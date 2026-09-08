#!/usr/bin/env python3
"""Single-request real-model ABBA timings with native selection receipts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import time
from urllib.error import URLError

from quactlize_gsm8k import request, save, digest, server_args, completion_payload
from quactlize_numerical import require, activity

PATTERN = r"(ffn_.*_exps|output\.weight)"


def timings(response, payload):
    require(
        response.get("tokens_evaluated") == len(payload["prompt"]),
        "prompt token count differs",
    )
    require(
        response.get("tokens_predicted") == payload["n_predict"],
        "generation count differs",
    )
    require(
        response.get("stop") is True and response.get("truncated") is False,
        "incomplete/truncated request",
    )
    t = response.get("timings", {})
    require(t.get("cache_n") == 0, "prompt KV reuse contaminates prefill")
    for name in ("prompt_n", "predicted_n", "prompt_ms", "predicted_ms"):
        require(
            type(t.get(name)) in (int, float)
            and math.isfinite(t[name])
            and t[name] > 0,
            "invalid server timer: " + name,
        )
    require(t["prompt_n"] == len(payload["prompt"]), "prompt timer denominator differs")
    require(
        t["predicted_n"] == payload["n_predict"], "decode timer denominator differs"
    )
    for name in ("n_predict", "temperature", "ignore_eos", "seed"):
        require(
            response.get("generation_settings", {}).get(name) == payload[name],
            "server settings differ: " + name,
        )
    return dict(
        prefill_us_per_token=1000 * t["prompt_ms"] / t["prompt_n"],
        decode_us_per_token=1000 * t["predicted_ms"] / t["predicted_n"],
        prefill_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
    )


def selection(text, manifest):
    require(
        not re.search(r"CUDA error:|PPU error:|GGML_ASSERT|GGML_ABORT", text),
        "runtime failure in model log",
    )
    modules = {m["key"]: m for m in manifest["modules"]}
    plans = []
    for line in text.splitlines():
        if "[quactlize-plan]" not in line:
            continue
        r = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
        require(
            r.get("op") in ("dense", "grouped")
            and r.get("route") in ("fq", "sf", "gemv"),
            "invalid plan receipt",
        )
        if r["route"] != "gemv":
            m = modules.get(r.get("build"))
            require(
                m and m["parent"]["symbol"] == r.get("parent"),
                "selected parent/build not in package",
            )
            p = m["parent"]
            require(
                int(r["q"]) == p["qtype"] and p["route"] == r["route"] + "-" + r["op"],
                "route and parent differ",
            )
            require(
                int(r["split"]) in (1, 2, 4, 8) and int(r["grid"]) >= 0,
                "invalid complete recipe",
            )
        plans.append(r)
    fallbacks = re.findall(r"\[quactlize\] ([^\n]*native policy miss[^\n]*)", text)
    prepass = [
        dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
        for line in text.splitlines()
        if "[quactlize-prepass]" in line
    ]
    return dict(
        plans=plans,
        fallbacks=fallbacks,
        prepass=prepass,
        fully_selected=not fallbacks
        and {r["op"] for r in plans} == {"dense", "grouped"},
    )


def run_arm(args, index, arm, tokens):
    label = f"{index}-{arm}"
    log_path = args.output / (label + ".log")
    command = server_args(
        args.binary,
        args.model,
        args.cache,
        "kpack" if arm == "native" else "reference",
        args.context,
        args.batch,
    )
    command[command.index("-ot") + 1] = (
        PATTERN + "=CUDA0" + ("_KPACK" if arm == "native" else "")
    )
    command += ["--no-warmup"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    if arm == "reference":
        env.pop("QUACTLIZE_KPACK_EXECUTION", None)
        env.pop("QUACTLIZE_KPACK_GEMV_POLICY", None)
    else:
        env["QUACTLIZE_KPACK_ROUTE"] = "auto"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = secrets.token_hex(24)
    alias = "native-" + secrets.token_hex(8)
    command += [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        alias,
        "--api-key",
        key,
    ]
    save(args.output / (label + ".command.json"), command[:-1] + ["<ephemeral-key>"])
    base = f"http://127.0.0.1:{port}"
    records = []
    begin = time.monotonic()
    with log_path.open("x") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            update = begin
            while True:
                require(proc.poll() is None, f"{label} server exited; see {log_path}")
                require(
                    time.monotonic() - begin < 900,
                    f"{label} startup exceeded 900 seconds",
                )
                try:
                    if request(base + "/health", key, timeout=1).get("status") == "ok":
                        break
                except (URLError, TimeoutError, ValueError):
                    pass
                if time.monotonic() - update > 15:
                    print(
                        f"KPACK_MODEL_LOADING arm={label} seconds={time.monotonic()-begin:.1f}",
                        flush=True,
                    )
                    update = time.monotonic()
                time.sleep(1)
            props = request(base + "/props", key)
            require(
                props.get("model_alias") == alias and props.get("total_slots") == 1,
                "wrong server or request concurrency",
            )
            require(
                props.get("default_generation_settings", {}).get("n_ctx")
                == args.context,
                "context differs",
            )
            save(args.output / (label + ".props.json"), props)
            if index:
                prior = json.loads((args.output / "0-reference.props.json").read_text())
                for f in ("model_path", "build_info"):
                    require(prior.get(f) == props.get(f), "A/B model/build differs")
            if tokens is None:
                text = (
                    "A short explanation of how matrix multiplication works with quantized weights. "
                    * 256
                )
                t = request(
                    base + "/tokenize",
                    key,
                    dict(content=text, add_special=True, parse_special=True),
                )["tokens"]
                require(len(t) >= max(args.prompts), "not enough prompt tokens")
                tokens = {str(n): t[:n] for n in args.prompts}
                require(
                    all(
                        type(x) is int and x >= 0
                        for row in tokens.values()
                        for x in row
                    ),
                    "invalid tokenization",
                )
                save(args.output / "input-tokens.json", tokens)
            startup = time.monotonic() - begin
            with (args.output / (label + ".jsonl")).open("x") as rows:
                for n in args.prompts:
                    for repeat in range(args.repeats + 1):
                        payload = completion_payload(
                            tokens[str(n)], 20260909, args.generate
                        )
                        payload["ignore_eos"] = True
                        tick = time.monotonic()
                        response = request(
                            base + "/completion", key, payload, timeout=600
                        )
                        rec = dict(
                            prompt=n,
                            repeat=repeat,
                            phase="first-use" if repeat == 0 else "steady",
                            request_sha256=digest(payload),
                            wall_seconds=time.monotonic() - tick,
                            response=response,
                        )
                        rows.write(json.dumps(rec, allow_nan=False) + "\n")
                        rows.flush()
                        rec["timings"] = timings(response, payload)
                        records.append(rec)
                        print(
                            f"KPACK_MODEL_PERF arm={label} prompt={n} repeat={repeat} "
                            f"prefill_us_per_token={rec['timings']['prefill_us_per_token']:.3f} "
                            f"decode_us_per_token={rec['timings']['decode_us_per_token']:.3f}",
                            flush=True,
                        )
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise ValueError(
                        f"{label} shutdown exceeded 60 seconds; logs preserved"
                    )
            save(
                args.output / (label + ".process.json"),
                dict(returncode=proc.returncode),
            )
    require(proc.returncode in (0, -15), f"{label} failed rc={proc.returncode}")
    text = log_path.read_text(errors="replace")
    if arm == "native":
        require("CUDA0_KPACK model buffer size" in text, "no K-pack placement")
        evidence = selection(text, args.manifest)
        require(evidence["plans"], "no native dispatch receipts")
    else:
        require(
            "CUDA0_KPACK model buffer size" not in text
            and "[quactlize-plan]" not in text,
            "reference contaminated",
        )
        evidence = {}
    save(args.output / (label + ".selection.json"), evidence)
    return (
        dict(
            arm=arm,
            index=index,
            startup_seconds=startup,
            selection=evidence,
            records=records,
        ),
        tokens,
    )


def summarize(arms, prompts, repeats):
    result = []
    require(
        [a["arm"] for a in arms] == ["reference", "native", "native", "reference"],
        "ABBA process denominator differs",
    )
    for n in prompts:
        groups = {}
        hashes = set()
        for arm in ("reference", "native"):
            groups[arm] = [
                r
                for a in arms
                if a["arm"] == arm
                for r in a["records"]
                if r["prompt"] == n and r["phase"] == "steady"
            ]
            require(len(groups[arm]) == 2 * repeats, "missing/duplicate timing samples")
            hashes.update(r["request_sha256"] for r in groups[arm])
        require(len(hashes) == 1, "A/B requests differ")
        row = dict(
            prompt_tokens=n,
            request_batch=1,
            samples_per_arm=2 * repeats,
            request_sha256=hashes.pop(),
        )
        for metric in ("prefill_us_per_token", "decode_us_per_token"):
            med = {
                arm: statistics.median(r["timings"][metric] for r in data)
                for arm, data in groups.items()
            }
            row[metric] = dict(
                **med, delta_pct=100 * (med["native"] / med["reference"] - 1)
            )
        # Generated continuations can diverge, so this is a real-model latency
        # comparison, not a paired identical-router microbenchmark.
        row["same_generated_text"] = (
            len(
                {r["response"].get("content") for data in groups.values() for r in data}
            )
            == 1
        )
        result.append(row)
    return result


def proof(args):
    log = args.output / "proof.log"
    report = args.output / "proof.asysrep"
    db = args.output / "proof.sqlite"
    command = [
        str(args.asys),
        "profile",
        "--trace",
        "hggc",
        "--hggc-trace-set",
        "kernel-activity",
        "--sample",
        "none",
        "--kill",
        "none",
        "--show-output",
        "true",
        "--output",
        str(report),
        str(args.proof_binary),
        "-m",
        str(args.model),
        "--mmap",
        "-ngl",
        "99",
        "--split-mode",
        "none",
        "--fit",
        "off",
        "--no-conversation",
        "--no-warmup",
        "--log-colors",
        "off",
        "--verbosity",
        "4",
        "-c",
        "512",
        "-b",
        "128",
        "-ub",
        "128",
        "-ot",
        PATTERN + "=CUDA0_KPACK",
        "--kpack-cache",
        str(args.cache),
        "-n",
        "8",
        "--temp",
        "0",
        "-p",
        "Explain matrix multiplication in one paragraph. " * 16,
    ]
    with log.open("x") as f:
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=900,
        )
    with (args.output / "proof-export.log").open("x") as f:
        subprocess.run(
            [str(args.asys), "export", "--output", str(db), str(report)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    plans = selection(log.read_text(errors="replace"), args.manifest)
    keys = {r["build"] for r in plans["plans"] if "build" in r}
    libraries = [args.bundle / "modules" / k / "kernel.so" for k in sorted(keys)] + [
        args.bundle / "libquactlize_ppu_execution.so"
    ]
    symbols = {}
    for library in libraries:
        listing = subprocess.check_output(
            [str(args.inspector), "--list-elf", str(library)], text=True
        )
        names = re.findall(r"^Func \d+:\s+(\S+)\s*$", listing, re.MULTILINE)
        require(names, "no PPU device symbols in " + str(library))
        decoded = subprocess.check_output(
            ["c++filt", "--no-recurse-limit"], input="\n".join(names) + "\n", text=True
        ).splitlines()
        require(len(decoded) == len(names), "demangled symbol count differs")
        for name, demangled in zip(names, decoded):
            if "cutlass::device_kernel<" in demangled or re.search(
                r"kpack_q(?:10|11|12|13|14)::", demangled
            ):
                item = symbols.setdefault(
                    name, dict(name=demangled, libraries=[], ops=[])
                )
                item["libraries"].append(str(library.relative_to(args.bundle)))
                if library.name == "kernel.so":
                    ops = {
                        r["op"]
                        for r in plans["plans"]
                        if r.get("build") == library.parent.name
                    }
                else:
                    q = re.search(r"kpack_q(\d+)::", demangled)
                    ops = {
                        r["op"]
                        for r in plans["plans"]
                        if r["route"] == "gemv" and q and r["q"] == q[1]
                    }
                item["ops"] = sorted(set(item["ops"]) | ops)
    total, matched = activity(db, symbols)
    require(matched, "no selected native compute kernel in PPU device trace")
    ops = {op for m in matched for op in symbols[m["mangled"]]["ops"]}
    require(
        ops == {"dense", "grouped"},
        "short proof lacks dense/grouped native compute activity: " + str(ops),
    )
    # A proof covers its own short request. It is not counted as an untraced
    # performance sample or as device evidence for every ABBA parent.
    result = dict(
        kernel_execution="PASS_SHORT_REQUEST",
        gpu_kernel_calls=total,
        matched=matched,
        observed_ops=sorted(ops),
        selection=plans,
        timing_scope="PROFILER_ONLY_NOT_PERFORMANCE",
        all_abba_parents_traced=False,
    )
    save(args.output / "proof.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "binary",
        "proof-binary",
        "model",
        "cache",
        "bundle",
        "output",
        "asys",
        "inspector",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--generate", type=int, default=128)
    p.add_argument("--prompts", type=int, nargs="+", default=[128, 512])
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    require(
        a.repeats >= 2
        and a.generate > 1
        and max(a.prompts) + a.generate < a.context
        and min(a.prompts) > 1,
        "invalid benchmark sizes",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    a.manifest = json.loads((a.bundle / "manifest.json").read_text())
    save(
        a.output / "protocol.json",
        dict(
            order=["reference", "native", "native", "reference"],
            request_batch=1,
            prompts=a.prompts,
            generate=a.generate,
            repeats=a.repeats,
            prefill_token_batch=a.batch,
            graphs=True,
            native_route="auto",
            dense_scope="Q6 output.weight; Q8_0 remains ordinary GPU",
            first_use_excluded_from_steady=True,
        ),
    )
    arms = []
    tokens = None
    for i, arm in enumerate(("reference", "native", "native", "reference")):
        result, tokens = run_arm(a, i, arm, tokens)
        arms.append(result)
    summary = dict(
        status="TIMINGS_COMPLETE_PROOF_PENDING",
        performance=summarize(arms, a.prompts, a.repeats),
        fully_selected=all(
            x["selection"]["fully_selected"] for x in arms if x["arm"] == "native"
        ),
        accuracy_admission="NOT_RETESTED",
        global_optimum="NOT_CLAIMED",
        kernel_execution="PENDING_SHORT_PROOF",
    )
    save(a.output / "summary.json", summary)
    proof_result = proof(a)
    summary.update(status="COMPLETE", kernel_execution=proof_result["kernel_execution"])
    save(a.output / "summary.json", summary)
    print("KPACK_MODEL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        raise SystemExit("KPACK_MODEL FAIL: " + str(e))
