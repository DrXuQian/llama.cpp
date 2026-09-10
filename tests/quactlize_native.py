#!/usr/bin/env python3
"""Single-request real-model ABBA timings with native selection receipts."""

import argparse
import copy
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

# The loader uses regex_search, and an explicit override bypasses automatic
# buffer admission. Match whole weight names, not attn_output.weight suffixes.
PATTERN = r"^(blk\.[0-9]+\.ffn_[a-z0-9_]+_exps\.weight|output\.weight)$"


def inventory_pattern(inventory):
    names = inventory.get("eligible")
    require(isinstance(names, list) and names
            and all(isinstance(name, str) and name for name in names)
            and len(set(names)) == len(names), "invalid trace tensor inventory")
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def validate_tokens(tokens, prompts):
    require(isinstance(tokens, dict) and set(tokens) == {str(n) for n in prompts}
            and all(isinstance(tokens[str(n)], list) and len(tokens[str(n)]) == n
                    and all(type(t) is int and t >= 0 for t in tokens[str(n)]) for n in prompts),
            "input token file does not match requested prompts")
    return tokens


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


def model_selection(args, text):
    keys = set()
    for line in text.splitlines():
        if "[quactlize-plan]" in line:
            keys.update(re.findall(r"\bbuild=([^\s]+)", line))
    require(all(re.fullmatch(r"[0-9a-f]{64}", k) for k in keys), "invalid selected build key")
    packaged = {m["key"]: m for m in args.manifest["modules"]}
    require(len(packaged) == len(args.manifest["modules"]), "duplicate package build key")
    modules = []
    for key in sorted(keys & packaged.keys()):
        m = packaged[key]
        relative = "modules/" + key + "/kernel.so"
        library = (args.bundle / relative).resolve(strict=True)
        require(m["path"] == relative and library.is_relative_to(args.bundle.resolve()),
                "selected package module escapes bundle")
        modules.append(m | dict(path=str(library), origin="package"))
    missing = keys - packaged.keys()
    if missing:
        require(args.jit_cache and args.jit_helper and args.jit_python,
                "selected parent/build absent from package and no JIT resolver configured")
        output = subprocess.check_output([
            str(args.jit_python), str(args.jit_helper), "inspect", "--cache", str(args.jit_cache),
            "--source-contract", args.manifest["jit_source_contract"], "--keys", *sorted(missing),
        ], text=True)
        cached = json.loads(output)["modules"]
        require(len(cached) == len(missing) and {m["key"] for m in cached} == missing,
                "cache inspection omitted or duplicated selected modules")
        for m in cached:
            path = (args.jit_cache / m["key"] / "kernel.so").resolve(strict=True)
            require(path == Path(m["path"]) and path.parent.parent == args.jit_cache.resolve(),
                    "selected JIT module escapes cache")
        modules.extend(m | dict(origin="jit-cache") for m in cached)
    for m in modules:
        with Path(m["path"]).open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == m["sha256"],
                    "selected module payload changed")
    evidence = selection(text, dict(modules=modules))
    evidence["modules"] = modules
    return evidence


class AsysSession:
    """Collect one completed request after the same process has warmed up."""
    def __init__(self, executable, output):
        self.executable = executable
        self.session = "kpack-proof-" + secrets.token_hex(8)
        self.report = output / "proof.asysrep"
        self.log = output / "proof-control.log"

    def command(self, application):
        return [str(self.executable), "launch", "--trace", "hggc", "--hggc-trace-set", "kernel-activity",
                "--sample", "none", "--wait", "primary", "--kill", "sigterm", "--show-output", "true",
                "--session-new", self.session, *application]

    def control(self, action, *options, check=True):
        with self.log.open("a") as log:
            subprocess.run([str(self.executable), action, "--session", self.session, *options],
                           stdout=log, stderr=subprocess.STDOUT, timeout=60, check=check)

    def start(self):
        self.control("start", "--output", str(self.report))

    def stop(self):
        self.control("stop")

    def close(self):
        # Only this uniquely named session; never stop another user's capture.
        self.control("shutdown", check=False)


def run_arm(args, index, arm, tokens, profile=None):
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
        getattr(args, "tensor_override_pattern", PATTERN)
        + "=CUDA0" + ("_KPACK" if arm == "native" else "")
    )
    command += ["--no-warmup"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    env.pop("GGML_CUDA_DISABLE_FUSION", None)
    if arm == "reference":
        env = {k: v for k, v in env.items() if not k.startswith("QUACTLIZE_KPACK_")}
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
    if profile:
        require(len(args.prompts) == 1 and args.repeats == 1, "trace needs one warmup and one captured request")
        command = profile.command(command)
    save(args.output / (label + ".command.json"), command[:-1] + ["<ephemeral-key>"])
    base = f"http://127.0.0.1:{port}"
    records = []
    begin = time.monotonic()
    phase = "startup-health"
    print(f"KPACK_MODEL_START arm={label} log={log_path}", flush=True)
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
            phase = "server-properties"
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
                phase = "tokenize"
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
            validate_tokens(tokens, args.prompts)
            save(args.output / "input-tokens.json", tokens)
            startup = time.monotonic() - begin
            with (args.output / (label + ".jsonl")).open("x") as rows:
                for n in args.prompts:
                    for repeat in range(args.repeats + 1):
                        phase = f"completion-prompt-{n}-repeat-{repeat}"
                        payload = completion_payload(
                            tokens[str(n)], 20260909, args.generate
                        )
                        payload["ignore_eos"] = True
                        if profile and repeat == 1:
                            profile.start()
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
                            f"phase={rec['phase']} included={int(repeat > 0 and profile is None)} "
                            f"prefill_us_per_token={rec['timings']['prefill_us_per_token']:.3f} "
                            f"decode_us_per_token={rec['timings']['decode_us_per_token']:.3f}",
                            flush=True,
                        )
                        if profile and repeat == 1:
                            profile.stop()
        except (ValueError, OSError) as error:
            status = proc.poll()
            save(
                args.output / (label + ".failure.json"),
                dict(
                    arm=label,
                    phase=phase,
                    error=str(error),
                    log=str(log_path),
                    returncode_before_cleanup=status,
                ),
            )
            raise ValueError(
                f"{label} phase={phase}: {error}; server_rc={status}; log={log_path}"
            ) from error
        finally:
            try:
                if profile:
                    # Session shutdown already signals the application. A second
                    # signal through the launcher interrupts graceful teardown.
                    profile.close()
                elif proc.poll() is None:
                    proc.terminate()
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        raise ValueError(
                            f"{label} shutdown exceeded 60 seconds; logs preserved"
                        )
            finally:
                save(
                    args.output / (label + ".process.json"),
                    dict(returncode=proc.returncode),
                )
    require(proc.returncode in (0, -15), f"{label} failed rc={proc.returncode}")
    text = log_path.read_text(errors="replace")
    if arm == "native":
        require("CUDA0_KPACK model buffer size" in text, "no K-pack placement")
        evidence = model_selection(args, text)
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
        for a in arms:
            require(sum(r["prompt"] == n and r["phase"] == "first-use" for r in a["records"]) == 1,
                    "each process/shape needs exactly one excluded first-use request")
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


def proof_parameters(args):
    prompt = getattr(args, "proof_prompt", 128)
    generate = getattr(args, "proof_generate", 8)
    require(type(prompt) is int and prompt > 1 and type(generate) is int and generate > 1,
            "invalid Asys prompt/generation lengths")
    return dict(context=max(512, ((prompt + generate) // 256 + 1) * 256),
                batch=prompt, generate=generate, prompts=[prompt], repeats=1)


def proof(args):
    report = args.output / "proof.asysrep"
    db = args.output / "proof.sqlite"
    capture = copy.copy(args)
    capture.output = args.output / "proof-request"
    capture.output.mkdir()
    for name, value in proof_parameters(args).items():
        setattr(capture, name, value)
    arm = getattr(args, "proof_arm", "native")
    source_tokens = getattr(args, "proof_tokens", None)
    tokens = (validate_tokens(json.loads(source_tokens.read_text()), capture.prompts)
              if source_tokens else None)
    proof_arm, tokens = run_arm(capture, 0, arm, tokens, profile=AsysSession(args.asys, args.output))
    with (args.output / "proof-export.log").open("x") as f:
        subprocess.run(
            [str(args.asys), "export", "--output", str(db), str(report)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    total, kernels = activity(db, None)
    save(args.output / "kernel-times.json", dict(
        arm=arm, gpu_kernel_calls=total, scope="ALL_CAPTURED_KERNELS_PROFILER_ONLY_NOT_WALL_LATENCY",
        sum_kernel_ns=sum(k["total_ns"] for k in kernels),
        kernels=sorted(kernels, key=lambda k: (-k["total_ns"], k["name"]))))
    captured = proof_arm["records"][-1]
    common = dict(
        arm=arm, gpu_kernel_calls=total, timing_scope="PROFILER_ONLY_NOT_PERFORMANCE",
        capture_scope="SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED",
        prompt_tokens=capture.prompts[0], generated_tokens=capture.generate,
        prefill_token_batch=capture.batch, input_tokens_sha256=digest(tokens),
        request_sha256=captured["request_sha256"],
        response_sha256=digest(captured["response"].get("content", "")))
    if arm == "reference":
        require(not any("quactlize" in k["name"] or re.search(r"kpack_q\d+::", k["name"])
                        for k in kernels), "reference trace contains K-pack execution")
        native_matvec = [k for k in kernels if re.search(r"\bmul_mat_(?:vec_q|vec_f|q)\s*<", k["name"])]
        require(native_matvec, "reference trace has no native llama matrix/vector kernel")
        result = common | dict(kernel_execution="REFERENCE_COMPUTE_OBSERVED", matched=native_matvec,
                               missing_ops=[], selection={}, all_abba_parents_traced=False)
        save(args.output / "proof.json", result)
        return result
    plans = proof_arm["selection"]
    libraries = [(Path(m["path"]), m["origin"] + "/" + m["key"], m["key"])
                 for m in plans["modules"]]
    libraries.append((args.bundle / "libquactlize_ppu_execution.so", "execution", None))
    symbols = {}
    for library, label, build in libraries:
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
                r"kpack_q(?:8|10|11|12|13|14)::", demangled
            ):
                item = symbols.setdefault(
                    name, dict(name=demangled, libraries=[], ops=[])
                )
                item["libraries"].append(label)
                if build:
                    ops = {
                        r["op"]
                        for r in plans["plans"]
                        if r.get("build") == build
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
    # A proof covers its own short request. It is not counted as an untraced
    # performance sample or as device evidence for every ABBA parent.
    result = dict(
        kernel_execution="PASS_SHORT_REQUEST" if ops == {"dense", "grouped"} else "PARTIAL_SHORT_REQUEST",
        gpu_kernel_calls=total,
        matched=matched,
        observed_ops=sorted(ops),
        missing_ops=sorted({"dense", "grouped"} - ops),
        selection=plans,
        timing_scope="PROFILER_ONLY_NOT_PERFORMANCE",
        capture_scope="SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED",
        prompt_tokens=capture.prompts[0],
        generated_tokens=capture.generate,
        prefill_token_batch=capture.batch,
        all_abba_parents_traced=False,
    )
    result.update(common)
    save(args.output / "proof.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "binary",
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
    p.add_argument("--proof-only", action="store_true", help="warmup and Asys capture without ABBA timings")
    p.add_argument("--proof-arm", choices=("native", "reference"), default="native",
                   help="trace route; reference retains the original llama CUDA kernels and fusions")
    p.add_argument("--proof-tokens", type=Path, help="reuse exact input-tokens.json from a matched trace")
    p.add_argument("--proof-prompt", type=int, default=128)
    p.add_argument("--proof-generate", type=int, default=8)
    p.add_argument("--tensor-inventory", type=Path, help="use the benchmark's exact eligible weight names")
    p.add_argument("--jit-cache", type=Path)
    p.add_argument("--jit-helper", type=Path)
    p.add_argument("--jit-python", type=Path)
    a = p.parse_args()
    require(a.proof_only or (a.proof_arm == "native" and a.proof_tokens is None),
            "proof arm/token overrides require --proof-only")
    a.tensor_override_pattern = (inventory_pattern(json.loads(a.tensor_inventory.read_text()))
                                 if a.tensor_inventory else PATTERN)
    capture_parameters = proof_parameters(a)
    require(
        a.proof_only or (a.repeats >= 2
        and a.generate > 1
        and max(a.prompts) + a.generate < a.context
        and min(a.prompts) > 1),
        "invalid benchmark sizes",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    a.manifest = json.loads((a.bundle / "manifest.json").read_text())
    if a.manifest.get("jit_required"):
        for name, value in (("EXECUTION", a.bundle), ("JIT_CACHE", a.jit_cache),
                            ("JIT_HELPER", a.jit_helper), ("JIT_PYTHON", a.jit_python)):
            configured = os.environ.get("QUACTLIZE_KPACK_" + name)
            require(value and configured and Path(configured).resolve() == value.resolve(),
                    "model/JIT evidence environment differs: " + name)
    save(
        a.output / "protocol.json",
        dict(
            order=[a.proof_arm] if a.proof_only else ["reference", "native", "native", "reference"],
            request_batch=1,
            prompts=capture_parameters["prompts"] if a.proof_only else a.prompts,
            generate=capture_parameters["generate"] if a.proof_only else a.generate,
            repeats=1 if a.proof_only else a.repeats,
            prefill_token_batch=capture_parameters["batch"] if a.proof_only else a.batch,
            proof_only=a.proof_only,
            proof_arm=a.proof_arm,
            proof_tokens=str(a.proof_tokens) if a.proof_tokens else None,
            proof_parameters=capture_parameters,
            graphs=True,
            native_route="auto",
            dense_scope="BENCHMARK_INVENTORY" if a.tensor_inventory else "Q6 output.weight; Q8_0 remains ordinary GPU",
            tensor_override_pattern=a.tensor_override_pattern,
            first_use_excluded_from_steady=True,
            first_use_scope="EACH_PROCESS_AND_PROMPT_SHAPE",
            asys_scope="SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED",
            jit_cache=str(a.jit_cache) if a.jit_cache else None,
            module_evidence="SELECTED_PAYLOAD_HASH_AND_SOURCE_BOUND_CACHE_RECEIPT",
        ),
    )
    if a.proof_only:
        result = proof(a)
        save(a.output / "summary.json", dict(status="TRACE_CAPTURED",
             kernel_execution=result["kernel_execution"], missing_native_ops=result["missing_ops"],
             performance="NOT_MEASURED", accuracy_admission="NOT_RETESTED"))
        print("KPACK_MODEL_TRACE " + json.dumps(result, sort_keys=True), flush=True)
        return
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
    complete = summary["fully_selected"] and proof_result["kernel_execution"] == "PASS_SHORT_REQUEST"
    summary.update(status="COMPLETE" if complete else "INCOMPLETE_NATIVE_COVERAGE",
                   kernel_execution=proof_result["kernel_execution"],
                   missing_native_ops=proof_result["missing_ops"])
    save(a.output / "summary.json", summary)
    print("KPACK_MODEL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    require(complete, "native coverage incomplete; timings, fallbacks and partial trace are preserved")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        raise SystemExit("KPACK_MODEL FAIL: " + str(e))
