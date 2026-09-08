#!/usr/bin/env python3
"""Summarize model numerics and executed kernels from Asight's SQLite export."""

import argparse
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import struct
import subprocess


FORMATS = {10: 2, 11: 3, 12: 0, 13: 1, 14: 4}
ROUTE = re.compile(r"\[ncp-route\]\s+MUL_MAT_ID\s+(\S+)\s+T=\s*(\d+).*->\s+(\S+)")
METRICS = {
    "ppl": r"Final estimate: PPL =\s*(\S+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*(\S+)",
    "mean_kld": r"Mean\s+KLD:\s*(\S+)",
    "max_kld": r"Maximum KLD:\s*(\S+)",
    "same_top_pct": r"Same top p:\s*(\S+)",
}
PHASES = ("reference-save", "reference-self", "kpack-save", "cache-self", "cache-reference", "cache-proof")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def gsm8k_records(path, limit=None):
    """Read local question/answer records without downloading a dataset."""
    require(limit is None or limit > 0, "GSM8K sample size must be positive")
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in islice((line for line in stream if line.strip()), limit)]
    elif suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(rows, list), "GSM8K JSON must be a list of question/answer objects")
        rows = rows[:limit]
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError("reading Parquet requires pyarrow; use a local JSONL export instead") from error
        with pq.ParquetFile(path) as table:
            batches = table.iter_batches(batch_size=limit or 1024, columns=["question", "answer"])
            rows = list(islice((row for batch in batches for row in batch.to_pylist()), limit))
    else:
        raise ValueError("GSM8K_FILE must be one local .jsonl, .json or .parquet file, not a directory")
    require(rows, "empty GSM8K sample")
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and all(isinstance(row.get(key), str) and row[key].strip()
                                             for key in ("question", "answer")),
                f"GSM8K record {index} needs nonempty question and answer strings")
    return rows


def gsm8k_corpus(path, limit=32):
    """Use a fixed local sample for likelihood comparison, not answer scoring."""
    return "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
                   for row in gsm8k_records(path, limit))


def inventory(bundle, inspector):
    """Read actual device entry points, not exported host wrappers."""
    kernels = {}
    libraries = {}
    for fmt in FORMATS.values():
        lib = bundle / f"libquactlize_ppu_fmt{fmt}.so"
        digest = hashlib.sha256()
        with lib.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        libraries[lib.name] = digest.hexdigest()
        output = subprocess.check_output([str(inspector), "--list-elf", str(lib)], text=True)
        names = re.findall(r"^Func \d+:\s+(\S+)\s*$", output, re.MULTILINE)
        require(names, f"no device symbols in {lib}")
        decoded = subprocess.check_output(["c++filt", "--no-recurse-limit"],
                                          input="\n".join(names) + "\n", text=True).splitlines()
        require(len(names) == len(decoded), "device symbol demangling changed the row count")
        selected = 0
        for mangled, demangled in zip(names, decoded):
            # Packing, metadata, gather/scatter and host launch stubs are not GEMM evidence.
            if "cutlass::device_kernel<" not in demangled or "GroupProblemShape<" not in demangled:
                continue
            selected += 1
            item = kernels.setdefault(mangled, {"name": demangled, "libraries": []})
            if lib.name not in item["libraries"]:
                item["libraries"].append(lib.name)
        require(selected, f"no grouped GEMM device entry points in {lib}")
    return {"libraries_sha256": libraries, "kernels": kernels}


def activity(db, symbols):
    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(HGPTI_ACTIVITY_KIND_KERNEL)")}
        require({"start", "end", "mangledName", "demangledName"} <= columns,
                "missing Asight device-kernel activity table/columns; API events are not execution evidence")
        by_demangled = {value["name"]: key for key, value in symbols.items()}
        rows = con.execute('''
            SELECT m.value, d.value, COUNT(*), SUM(k.end-k.start), MIN(k.end-k.start)
            FROM HGPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds m ON k.mangledName=m.id
            LEFT JOIN StringIds d ON k.demangledName=d.id
            GROUP BY m.value, d.value
        ''')
        total = 0
        matched = []
        for mangled, demangled, count, duration, minimum in rows:
            require(minimum is not None and minimum > 0, "incomplete/nonpositive kernel activity duration")
            total += count
            key = mangled if mangled in symbols else by_demangled.get(demangled)
            if key:
                matched.append({"mangled": key, "name": symbols[key]["name"],
                                "libraries": symbols[key]["libraries"], "calls": count,
                                "total_ns": duration})
        require(total > 0, "empty GPU kernel trace")
        return total, matched
    finally:
        con.close()


def logprobs(path, expected_context=256, expected_chunks=2):
    with path.open("rb") as stream:
        require(stream.read(8) == b"_logits_", "invalid log-probability header")
        context, vocab, chunks = struct.unpack("<iii", stream.read(12))
        require(context == expected_context and chunks == expected_chunks and context > 1 and chunks > 0
                and vocab > 0, "unexpected evaluation coverage")
        tokens = stream.read(context * chunks * 4)
    scored = chunks * (context - 1 - context // 2)
    size = 20 + context * chunks * 4 + scored * (2 * ((vocab + 1) // 2) + 4) * 2
    require(path.stat().st_size == size, "truncated or unexpected log-probability payload")
    return {"context": context, "chunks": chunks, "vocab": vocab, "scored_tokens": scored,
            "tokens_sha256": hashlib.sha256(tokens).hexdigest()}


def analyze_log(log, manifest, batch, phase, context=256, chunks=2, placement_only=False):
    require(batch in (1, 128) and context > 1 and context >= batch and context % batch == 0 and chunks > 0,
            "invalid evaluation geometry")
    baseline = phase.startswith("reference")
    cached = phase.startswith("cache")
    text = log.read_text(errors="replace")
    require(not re.search(r"CUDA error:|PPU error:|failed to decode|failed reading log-probs", text),
            "model evaluation reported a runtime error")
    ppl = phase.endswith(("save", "proof", "perf"))
    label = "perplexity: calculating perplexity" if ppl else "kl_divergence: computing"
    start = re.search(re.escape(label) + rf" over {chunks} chunks, n_ctx={context}, batch_size={batch}, n_seq=1\b", text)
    if start is None:
        observed = re.findall(r"^.*(?:perplexity:|kl_divergence:).*chunks.*$", text, re.MULTILINE)[-3:]
        raise ValueError("evaluation did not start with the requested context/chunks/batch; "
                         f"observed={observed!r}")
    expected = {t["name"] for t in manifest["tensors"] if t.get("route_class") == "grouped"}
    require(expected, "cache manifest contains no grouped tensors")
    routed = {name for name, tokens, route in ROUTE.findall(text[start.end():])
              if int(tokens) == batch and route == "so-quactlize-kpack"}
    if baseline:
        require("so-quactlize-kpack" not in text, "ordinary GPU baseline entered the K-pack route")
        require(not re.search(r"CUDA\d+_KPACK model buffer size", text), "ordinary GPU baseline placed K-pack weights")
    elif placement_only:
        placed = re.search(r"CUDA\d+_KPACK model buffer size\s*=\s*(\S+) MiB", text)
        require(placed and math.isfinite(float(placed[1])) and float(placed[1]) > 0, "missing K-pack buffer placement")
    else:
        require(routed == expected, f"grouped route coverage differs: missing={sorted(expected-routed)} "
                f"extra={sorted(routed-expected)}")
    if cached:
        require(re.search(rf"cache_uploads={len(expected)} resident_misses=0\b", text), "not a full cache hit")
        require("GPU pack queued" not in text, "cache comparison repacked weights")
    metrics = {}
    required = ("ppl",) if ppl else ("mean_kld", "max_kld", "ppl_ratio", "same_top_pct")
    for key, regex in METRICS.items():
        match = re.search(regex, text)
        if match:
            value = float(match[1])
            require(math.isfinite(value), f"nonfinite {key}")
            metrics[key] = value
    require(all(key in metrics for key in required), "incomplete numerical results")
    return {"phase": phase, "batch": batch, "context": context, "chunks": chunks,
            "route_verdict": "PASS", "gpu_kernel_calls": None, "quactlize_grouped_gemm_calls": None,
            "routed_tensors": None if placement_only else len(routed), "metrics": metrics, "kernels": None,
            "route_evidence": "BUFFER_PLACEMENT" if placement_only else "POST_START_ROUTE_LOG",
            "kernel_execution": "NOT_COLLECTED", "accuracy_admission": "PENDING_REVIEW"}


def analyze(log, db, index, manifest, batch, phase, context=256, chunks=2):
    result = analyze_log(log, manifest, batch, phase, context, chunks)
    baseline = phase.startswith("reference")
    expected = {t["name"] for t in manifest["tensors"] if t.get("route_class") == "grouped"}
    total, matched = activity(db, index["kernels"])
    calls = sum(item["calls"] for item in matched)
    require((calls == 0) if baseline else (calls > 0),
            "device kernel execution contradicts the requested route")
    if not baseline:
        # The current grouped device ABI enqueues one GEMM per tensor and batch.
        # A single captured/warmup launch is not coverage of the full evaluation.
        minimum_calls = len(expected) * (context * chunks // batch)
        require(calls >= minimum_calls,
                f"incomplete evaluation kernel trace: {calls} GEMMs, need at least {minimum_calls}")
        seen_libs = {lib for item in matched for lib in item["libraries"]}
        need_libs = {f"libquactlize_ppu_fmt{FORMATS[t['ggml_type']]}.so" for t in manifest["tensors"]
                     if t["name"] in expected}
        require(need_libs <= seen_libs, f"missing model-format GEMM evidence: {sorted(need_libs-seen_libs)}")
    result.update(gpu_kernel_calls=total, quactlize_grouped_gemm_calls=calls,
                  kernels=matched, kernel_execution="PASS")
    return result


def performance(log, manifest, batch, phase, context=1024, chunks=8):
    """Read model evaluation timers, not profiler, loading or probability-file wall time."""
    require(phase in ("reference-perf", "cache-perf"), "unexpected performance phase")
    result = analyze_log(log, manifest, batch, phase, context, chunks, placement_only=True)
    text = log.read_text(errors="replace")
    require("warming up the model with an empty run" in text, "performance warmup is missing")
    require("[ncp-route]" not in text and " D " not in text, "performance log has verbose debug tracing")
    require("saving all logits" not in text and "kl_divergence:" not in text, "performance wrote/compared probabilities")
    timings = {}
    for name, pattern in (("prompt", r"prompt eval time\s*=\s*(\S+) ms /\s*(\d+) tokens"),
                          ("decode", r":\s+eval time\s*=\s*(\S+) ms /\s*(\d+) runs")):
        matches = re.findall(pattern, text)
        require(len(matches) == 1, f"missing/duplicate {name} evaluation timer (matches={len(matches)}); "
                "library INFO timers need --verbosity 4, not 3")
        ms, count = float(matches[0][0]), int(matches[0][1])
        require(math.isfinite(ms) and ms >= 0, f"invalid {name} evaluation timer")
        timings[name] = {"ms": ms, "tokens": count}
    prompt, decode = timings["prompt"], timings["decode"]
    scored = chunks * (context - 1 - context // 2)
    if batch == 128:
        require(prompt["tokens"] == chunks*context and decode == {"ms": 0.0, "tokens": 1},
                "prefill timer coverage differs")
        selected = prompt
    else:
        require(decode["tokens"] == scored and prompt["tokens"] == chunks*context-scored,
                "decode timer coverage differs")
        selected = decode
    require(prompt["ms"] > 0 and selected["ms"] > 0, "empty timed evaluation")
    result["performance"] = {"scope": "MODEL_EVAL_TIMERS", "profiler": False, "timings": timings,
                             "tokens_per_second": selected["tokens"]*1000/selected["ms"]}
    return result


def performance_summary(records):
    require(all(r["route_verdict"] == "PASS" and r["kernel_execution"] == "NOT_COLLECTED"
                and r["performance"]["profiler"] is False for r in records), "invalid performance evidence")
    batches = sorted({r["batch"] for r in records})
    require(batches and all(b in (1, 128) for b in batches), "invalid performance batches")
    rows = []
    for batch in batches:
        arms = {}
        for phase in ("reference-perf", "cache-perf"):
            samples = [r for r in records if r["batch"] == batch and r["phase"] == phase]
            require(len(samples) == 2, "performance needs two samples per arm")
            require(len({(r["context"], r["chunks"]) for r in records if r["batch"] == batch}) == 1,
                    "performance samples have different coverage")
            speeds = [r["performance"]["tokens_per_second"] for r in samples]
            require(all(math.isfinite(v) and v > 0 for v in speeds), "invalid performance speed")
            arms[phase] = {"samples_tps": speeds, "median_tps": statistics.median(speeds),
                           "min_tps": min(speeds), "max_tps": max(speeds),
                           "spread_pct": 100*(max(speeds)-min(speeds))/statistics.median(speeds)}
        rows.append({"batch": batch, "arms": arms,
                     "kpack_delta_pct": 100*(arms["cache-perf"]["median_tps"] /
                                             arms["reference-perf"]["median_tps"] - 1)})
    require(len(records) == 4*len(batches), "unexpected performance records")
    return {"order": "reference,cache,cache,reference", "rows": rows, "admission": "PENDING_REVIEW"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("bundle", type=Path)
    inv.add_argument("inspector", type=Path)
    for name in ("check", "check-log", "performance"):
        check = sub.add_parser(name)
        keys = ("log", "db", "index", "manifest") if name == "check" else ("log", "manifest")
        for key in keys:
            check.add_argument(key, type=Path)
        check.add_argument("--batch", type=int, choices=(1, 128), required=True)
        check.add_argument("--phase", choices=("reference-perf", "cache-perf") if name == "performance" else PHASES,
                           required=True)
        check.add_argument("--context", type=int, default=256)
        check.add_argument("--chunks", type=int, default=2)
    lp = sub.add_parser("logprobs")
    lp.add_argument("path", type=Path)
    lp.add_argument("--context", type=int, default=256)
    lp.add_argument("--chunks", type=int, default=2)
    corpus = sub.add_parser("gsm8k")
    corpus.add_argument("path", type=Path)
    corpus.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    try:
        if args.command == "gsm8k":
            print(gsm8k_corpus(args.path, args.limit), end="")
            return
        if args.command == "inventory":
            result = inventory(args.bundle, args.inspector)
        elif args.command == "logprobs":
            result = logprobs(args.path, args.context, args.chunks)
        elif args.command in ("check-log", "performance"):
            fn = performance if args.command == "performance" else analyze_log
            result = fn(args.log, json.loads(args.manifest.read_text()), args.batch, args.phase, args.context, args.chunks)
        else:
            result = analyze(args.log, args.db, json.loads(args.index.read_text()),
                             json.loads(args.manifest.read_text()), args.batch, args.phase, args.context, args.chunks)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, sqlite3.Error, subprocess.CalledProcessError, struct.error) as error:
        parser.exit(1, f"KPACK_NUMERICAL FAIL: {error}\n")


if __name__ == "__main__":
    main()
