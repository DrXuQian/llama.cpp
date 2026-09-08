#!/usr/bin/env python3
"""Single-request, generated-answer comparison against the ordinary GPU route."""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from quactlize_numerical import gsm8k_records, require


PROMPT = ("{question}\n\nSolve the problem step by step. Finish with a separate line "
          "in exactly this format: #### <number>. Use only the final numeric answer "
          "after ####, without units or other text.")
NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)")
OPENER = build_opener(ProxyHandler({}))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def number(text):
    text = text.strip()
    if len(text) > 128 or not NUMBER.fullmatch(text):
        return None
    value = Decimal(text.replace(",", ""))
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def final_answer(text):
    """Do not mistake a number in the reasoning for a final answer."""
    lines = text.strip().splitlines()
    if not lines or not lines[-1].strip().startswith("####"):
        return None
    return number(lines[-1].strip()[4:])


def prepare(source, count, seed):
    rows = gsm8k_records(source)
    require(0 < count <= len(rows), f"requested {count} questions but dataset has {len(rows)}")
    cases = []
    for index in random.Random(seed).sample(range(len(rows)), count):
        row = rows[index]
        gold = final_answer(row["answer"])
        require(gold is not None, f"invalid GSM8K gold answer at row {index}")
        cases.append({"id": index, "question": row["question"], "gold": gold,
                      "prompt": PROMPT.format(question=row["question"])})
    return cases


def request(url, key, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    try:
        with OPENER.open(req, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        raise ValueError(f"HTTP {error.code}: {error.read(4096).decode(errors='replace')}") from error
    require(isinstance(result, dict) and "error" not in result, f"invalid server response: {result}")
    return result


def server_args(binary, model, cache, arm, context, batch):
    args = [str(binary), "-m", str(model), "--mmap", "-ngl", "99", "--split-mode", "none",
            "--fit", "off", "-c", str(context), "-b", str(batch), "-ub", str(batch),
            "-np", "1", "-t", "16", "-tb", "32", "--threads-http", "2",
            "--jinja", "--reasoning", "off", "--no-context-shift", "--cache-ram", "0",
            "--no-webui", "--log-colors", "off", "--verbosity", "4",
            "-ot", "ffn_.*_exps=CUDA0" + ("_KPACK" if arm == "kpack" else "")]
    if arm == "kpack":
        args += ["--kpack-cache", str(cache)]
    return args


def route_receipt(text, manifest, arm):
    require(not re.search(r"CUDA error:|PPU error:|GGML_ASSERT|grouped launch returned", text),
            "server reported a runtime error")
    placements = re.findall(r"CUDA[0-9]+_KPACK model buffer size\s*=\s*([0-9.]+) MiB", text)
    if arm == "reference":
        require(not placements and "so-quactlize-kpack" not in text,
                "ordinary GPU reference used K-pack")
        require(re.search(r"CUDA[0-9]+ model buffer size\s*=\s*[1-9][0-9.]* MiB", text),
                "missing ordinary GPU placement")
    else:
        expected = [t["name"] for t in manifest["tensors"] if t.get("route_class") == "grouped"]
        require(expected and len(set(expected)) == len(expected), "invalid grouped cache manifest")
        require(placements and sum(map(float, placements)) > 0, "missing K-pack placement")
        require(re.search(rf"cache_uploads={len(expected)} resident_misses=0\b", text),
                "K-pack arm did not fully hit the existing disk cache")
        require("GPU pack queued" not in text, "K-pack arm repacked weights")
    return {"route": arm, "placement": "PASS", "kernel_execution": "NOT_COLLECTED",
            "evidence": "BUFFER_PLACEMENT_AND_COMPLETED_REQUESTS"}


def completion_payload(tokens, seed, max_tokens):
    return {"prompt": tokens, "id_slot": 0, "stream": False, "cache_prompt": False,
            "seed": seed, "n_predict": max_tokens, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
            "mirostat": 0, "dynatemp_range": 0.0, "ignore_eos": False}


def score(response, payload, gold):
    require(response.get("stop") is True and response.get("id_slot") == 0,
            "incomplete response or wrong slot")
    require(isinstance(response.get("content"), str), "missing generated text")
    require(response.get("tokens_evaluated") == len(payload["prompt"]), "prompt token count differs")
    require(response.get("timings", {}).get("cache_n") == 0, "unexpected prompt KV cache reuse")
    for key in ("seed", "n_predict", "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
                "presence_penalty", "frequency_penalty", "dry_multiplier", "mirostat", "dynatemp_range", "ignore_eos"):
        require(response.get("generation_settings", {}).get(key) == payload[key],
                f"server generation setting differs: {key}")
    count = response.get("tokens_predicted")
    require(type(count) is int and 0 < count <= payload["n_predict"], "invalid generated token count")
    require(response.get("stop_type") in ("eos", "word", "limit"), "unknown stop type")
    require(type(response.get("truncated")) is bool, "missing context truncation status")
    answer = final_answer(response["content"])
    truncated = response["truncated"] or response["stop_type"] == "limit"
    return {"answer": answer, "correct": not truncated and answer is not None and answer == gold,
            "parse_failed": answer is None, "truncated": truncated}


def run_arm(args, arm, cases, manifest):
    root = args.output
    log_path = root / f"{arm}.server.log"
    command = server_args(args.binary, args.model, args.cache, arm, args.context, args.batch)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    with (root / f"{arm}.cli.log").open("w") as log:
        subprocess.run(command + ["--help"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    alias, key = "gsm8k-" + secrets.token_hex(8), secrets.token_hex(24)
    command += ["--host", "127.0.0.1", "--port", str(port), "--alias", alias, "--api-key", key]
    save(root / f"{arm}.command.json", command[:-1] + ["<ephemeral-local-key>"])
    base = f"http://127.0.0.1:{port}"
    print(f"KPACK_GSM8K_START arm={arm} questions={len(cases)} requests_in_flight=1", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            next_update = started + 15
            while True:
                require(process.poll() is None, f"{arm} server exited rc={process.returncode}; see {log_path}")
                require(time.monotonic() - started < 600, f"{arm} server startup exceeded 600 seconds")
                try:
                    if request(base + "/health", key, timeout=1).get("status") == "ok":
                        break
                except (URLError, TimeoutError, ValueError):
                    pass
                if time.monotonic() >= next_update:
                    print(f"KPACK_GSM8K_LOADING arm={arm} seconds={time.monotonic()-started:.0f}", flush=True)
                    next_update = time.monotonic() + 15
                time.sleep(1)
            props = request(base + "/props", key)
            require(props.get("model_alias") == alias, "connected server is not the process just launched")
            require(props.get("total_slots") == 1, "business request batch must be one")
            require(props.get("default_generation_settings", {}).get("n_ctx") == args.context,
                    "server context differs")
            require(props.get("chat_template"), "model has no chat template")
            save(root / f"{arm}.props.json", props)
            if arm == "kpack":
                prior = json.loads((root / "reference.props.json").read_text())
                for field in ("chat_template", "model_path", "build_info"):
                    require(prior.get(field) == props.get(field), f"A/B server {field} differs")
            route_receipt(log_path.read_text(errors="replace"), manifest, arm)
            elapsed = time.monotonic()
            previous = {}
            if arm == "kpack":
                previous = {r["id"]: r for r in read_rows(root / "reference.jsonl")}
            with (root / f"{arm}.jsonl").open("x") as output:
                for i, case in enumerate(cases):
                    require(process.poll() is None, f"{arm} server stopped before question {case['id']}")
                    # Only the question and fixed instruction cross the HTTP boundary, never the gold solution.
                    rendered = request(base + "/apply-template", key, {
                        "messages": [{"role": "user", "content": case["prompt"]}],
                        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
                    tokens = request(base + "/tokenize", key, {
                        "content": rendered, "add_special": True, "parse_special": True})["tokens"]
                    require(tokens and all(type(t) is int and t >= 0 for t in tokens), "invalid input tokens")
                    require(len(tokens) + args.max_tokens < args.context, "prompt plus generation cap exceeds context")
                    payload = completion_payload(tokens, args.seed + i, args.max_tokens)
                    input_hash = digest(payload)
                    if arm == "kpack":
                        require(previous[case["id"]]["request_sha256"] == input_hash,
                                f"A/B input tokens/settings differ for question {case['id']}")
                    tick = time.monotonic()
                    response = request(base + "/completion", key, payload)
                    row = {"id": case["id"], "request_sha256": input_hash,
                           "input_tokens": tokens, "rendered_prompt": rendered,
                           "wall_seconds": time.monotonic() - tick, "response": response}
                    # Preserve raw responses even if receipt validation fails.
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    grade = score(response, payload, case["gold"])
                    remaining = (time.monotonic() - elapsed) / (i + 1) * (len(cases) - i - 1) / 60
                    print(f"KPACK_GSM8K_PROGRESS arm={arm} completed={i+1}/{len(cases)} "
                          f"correct={int(grade['correct'])} truncated={int(grade['truncated'])} "
                          f"parse_failed={int(grade['parse_failed'])} remaining_arm_minutes={remaining:.1f}", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise ValueError(f"{arm} server needed forced shutdown; logs preserved")
            save(root / f"{arm}.process.json", {"returncode": process.returncode})
    require(process.returncode in (0, -15), f"{arm} server failed rc={process.returncode}")
    return route_receipt(log_path.read_text(errors="replace"), manifest, arm)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root, cases, seed, max_tokens):
    arms = {}
    for arm in ("reference", "kpack"):
        records = read_rows(root / f"{arm}.jsonl")
        require([r["id"] for r in records] == [c["id"] for c in cases], f"{arm} incomplete/duplicate/out-of-order results")
        grades = []
        for i, (case, row) in enumerate(zip(cases, records)):
            payload = completion_payload(row["input_tokens"], seed + i, max_tokens)
            require(digest(payload) == row["request_sha256"], "saved request digest differs")
            grades.append(score(row["response"], payload, case["gold"]))
        arms[arm] = {"records": records, "grades": grades}
    pairs = []
    for i, case in enumerate(cases):
        require(arms["reference"]["records"][i]["request_sha256"] == arms["kpack"]["records"][i]["request_sha256"],
                "A/B input token or generation setting mismatch")
        pairs.append({"id": case["id"], "gold": case["gold"],
                      **{arm: arms[arm]["grades"][i] for arm in arms}})
    counts = {}
    for arm in arms:
        grades = arms[arm]["grades"]
        correct = sum(g["correct"] for g in grades)
        counts[arm] = {"questions": len(cases), "correct": correct, "accuracy_pct": 100 * correct / len(cases),
                       "parse_failed": sum(g["parse_failed"] for g in grades),
                       "truncated": sum(g["truncated"] for g in grades)}
    return {"status": "COMPLETE", "accuracy_admission": "PENDING_REVIEW", "request_batch": 1,
            "kernel_execution": "NOT_COLLECTED", "scope": "PAIRED_ZERO_SHOT_FIXED_SAMPLE",
            "arms": counts, "accuracy_delta_pp": counts["kpack"]["accuracy_pct"] - counts["reference"]["accuracy_pct"],
            "reference_correct_kpack_wrong": [p["id"] for p in pairs if p["reference"]["correct"] and not p["kpack"]["correct"]],
            "reference_wrong_kpack_correct": [p["id"] for p in pairs if not p["reference"]["correct"] and p["kpack"]["correct"]],
            "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128, choices=(1, 128))
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.cache = args.cache.resolve(strict=True)
    require(0 < args.max_tokens < args.context and 0 <= args.seed < 2**32 - args.cases, "invalid token cap or seed")
    require(args.output.is_dir() and not (args.output / "protocol.json").exists(), "output must be a fresh result directory")
    cases = prepare(args.dataset, args.cases, args.seed)
    save(args.output / "protocol.json", {"version": 1, "source_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "sampling_seed": args.seed, "context": args.context, "prefill_token_batch": args.batch,
        "request_batch": 1, "max_generated_tokens": args.max_tokens, "thinking": False,
        "prompt_template": PROMPT, "selection": "random.Random.sample without replacement", "cases": cases})
    manifest = json.loads((args.cache / "manifest.json").read_text())
    receipts = {}
    for arm in ("reference", "kpack"):
        receipts[arm] = run_arm(args, arm, cases, manifest)
        save(args.output / f"{arm}.route.json", receipts[arm])
    result = summarize(args.output, cases, args.seed, args.max_tokens)
    result["routes"] = receipts
    save(args.output / "summary.json", result)
    print("KPACK_GSM8K_SUMMARY " + json.dumps({k: v for k, v in result.items() if k != "pairs"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(f"KPACK_GSM8K FAIL: {error}") from error
