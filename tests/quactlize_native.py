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
import sys
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


def measured_decode_recipe(name):
    match = re.search(r"quactlize::execution::simt::measured_decode_kernel<\s*" +
                      r",\s*".join([r"(\d+)"] * 14) + r",\s*(true|false|0|1),\s*(true|false|0|1)\s*>", name)
    if not match:
        return None
    v = tuple(map(int, match.groups()[:14]))
    q, mode, n, k, experts, topk, channels, compute, variant, columns, warps, values, split, changes = v
    if (q not in (8, 10, 11, 12, 13, 14) or mode not in (0, 2) or min(n, k) <= 0 or
            n % 256 or k % 256 or compute not in (0, 1) or changes not in (0, 1, 3) or
            columns not in (4, 8) or warps not in (2, 4, 8) or values not in (2, 4, 8) or
            split not in (1, 2, 4, 8) or not 0 <= variant <= (5 if q == 8 else 3) or
            (mode == 0 and (experts, topk, channels) != (1, 1, 1)) or
            (mode == 2 and (experts != 256 or topk != 8 or channels not in (1, 8)))):
        return None
    return v


def measured_decode_matches_plan(recipe, plan):
    if recipe is None:
        return True
    q, mode, n, k, experts, topk, channels, compute, variant, columns, warps, values, split, changes = recipe
    return (plan.get('op') == ('dense' if mode == 0 else 'grouped') and
            all(int(plan.get(field, -1)) == value for field, value in
                (('q', q), ('n', n), ('k', k), ('rows', topk), ('split', split))) and
            int(plan.get('activation') == 'BF16') == compute)


def simt_symbol_recipe(name):
    measured = measured_decode_recipe(name)
    if measured:
        q, mode, n, k, experts, topk, channels, compute, variant, columns, warps, values, split, changes = measured
        return (q, 1, variant, columns, warps, values, compute)
    fixed = re.search(r"quactlize::execution::simt::q8_vector::kernel_model<\s*" +
                      r",\s*".join([r"(\d+)"] * 6) + r",\s*(true|false|0|1),\s*(\d+),\s*(\d+),\s*(\d+)\s*>", name)
    if fixed:
        storage, compute, variant, columns, warps, values = map(int, fixed.groups()[:6])
        hoist = fixed.group(7) in ("true", "1")
        if (storage, compute, variant, columns, warps, values, hoist, *map(int, fixed.groups()[7:])) not in (
                (1, 0, 1, 8, 4, 4, False, 2048, 4096, 8), (1, 0, 1, 8, 4, 4, True, 8192, 2048, 1)):
            return None
        return (8, storage, variant + 4, columns, warps, values, compute)
    single = re.search(r"quactlize::execution::simt::q8_vector::kernel_s1<\s*" +
                       r",\s*".join([r"(\d+)"] * 6) + r",\s*(true|false|0|1)\s*>", name)
    if single:
        recipe = tuple(map(int, single.groups()[:6]))
        if (*recipe, single.group(7) in ("true", "1")) not in (
                (1, 0, 1, 4, 2, 4, True), (1, 0, 1, 4, 8, 4, False)):
            return None
        storage, compute, variant, columns, warps, values = recipe
        return (8, storage, variant + 4, columns, warps, values, compute)
    vector = re.search(r"quactlize::execution::simt::q8_vector::kernel<\s*" + r",\s*".join([r"(\d+)"] * 6) + r"(?:,\s*(?:true|false|0|1))?\s*>", name)
    if vector:
        storage, compute, variant, columns, warps, values = map(int, vector.groups())
        return (8, storage, variant + 4, columns, warps, values, compute)
    match = re.search(r"quactlize::execution::simt::register_reuse(_model)?<\s*(\d+(?:\s*,\s*\d+)*)\s*>", name)
    if not match:
        return None
    values = tuple(map(int, match[2].split(',')))
    if len(values) not in ((8, 10) if match[1] else (6, 7, 8)):
        return None
    if len(values) >= 8 and values[7] not in (0, 1, 3):
        return None
    if len(values) == 10 and min(values[8:]) <= 0:
        return None
    return (*values[:6], values[6] if len(values) > 6 else 0)


def q4_symbol_recipe(name):
    match = re.search(r"quactlize::execution::q4_decode::kernel(_bf16)?<\s*" +
                      r",\s*".join([r"(\d+)"] * 8) + r"\s*>", name)
    return (*map(int, match.groups()[1:]), int(bool(match[1]))) if match else None


def q4_symbol_matches_plan(recipe, plan):
    if not recipe or plan.get("route") != "gemv-q4-s1":
        return False
    fields = ("reader", "variant", "warps", "values", "columns", "n", "k")
    return recipe == (1, *[int(plan.get(k, -1)) for k in fields], int(plan.get("activation") == "BF16"))


def paired_symbol_recipe(name):
    if re.search(r"quactlize::fusion::simt_gate_up_q8_tile16\(", name):
        return dict(q=8, storage=1, compute=0, warps=8, tile_m=0, backend='simt', measured_tile16=True)
    simt=re.search(r"quactlize::fusion::simt_gate_up(?:_model)?<\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+),\s*(\d+))?\s*>",name)
    if simt:
        q,storage,compute,warps=map(int,simt.groups()[:4])
        result=dict(q=q,storage=storage,compute=compute,warps=warps,tile_m=0,backend='simt')
        if simt[5] is not None:result.update(physical_n=int(simt[5]),k=int(simt[6]))
        return result
    tc=re.search(r"quactlize::fusion::tc_gate_up<quactlize::fusion::TcTypes<\s*(\d+),\s*(\d+),\s*cutlass::(half_t|bfloat16_t),\s*float\s*>",name)
    if tc:
        return dict(q=int(tc[1]),storage=1,compute=int(tc[3]=='bfloat16_t'),warps=0,tile_m=int(tc[2]),backend='tc')
    return None


def paired_matches_plan(recipe,plan):
    if not recipe:return False
    recipe=dict(recipe)
    if recipe.pop('measured_tile16', False):
        if int(plan.get('tokens',0))!=1 or (int(plan['n']),int(plan['k'])) not in ((512,2048),(1024,3072)):
            return False
    if 'physical_n' in recipe:
        if recipe.pop('physical_n')!=2*int(plan['n']) or recipe.pop('k')!=int(plan['k']):return False
    return recipe==dict(q=int(plan['q']),storage=1,compute=int(plan['activation']=='BF16'),
        warps=int(plan['warps']),tile_m=int(plan['tile_m']),backend=plan['backend'])


def paired_selection(text,manifest):
    records=[]
    for line in text.splitlines():
        if '[quactlize-paired-plan]' not in line:continue
        r=dict(re.findall(r'([a-z_]+)=([^\s]+)',line))
        receipt=manifest.get('paired_gate_up',{})
        require(receipt.get('layout_id')=='0x47554e3400000001' and r.get('layout')==receipt['layout_id'],
                'unbound paired layout receipt')
        q,tokens,experts=map(lambda key:int(r.get(key,0)),('q','tokens','experts'))
        old_shape=r.get('n')=='512' and r.get('k')=='2048' and 1<=tokens<=8
        tp2_shared=(q==8 and r.get('n')=='1024' and r.get('k')=='3072' and tokens==1 and
                    receipt.get('q8_shared')=='F16_N512_K2048_T1_8_N1024_K3072_T1')
        require(old_shape or tp2_shared,'unmeasured paired shape')
        if q==8:
            require(r.get('op')=='dense' and experts==1 and r.get('activation')=='FP16','paired shared precision/scope differs')
            expected=('simt',1,0,8)
        else:
            require(q==12 and r.get('op')=='grouped' and experts==256 and r.get('activation')=='BF16',
                    'paired routed precision/scope differs')
            expected=(('simt',1,0,4 if tokens==2 else 8) if tokens<=2 or tokens==4
                      else ('tc',2 if tokens==3 else 1,16 if tokens<=6 else 8,0))
        require((r.get('backend'),int(r.get('split',0)),int(r.get('tile_m',-1)),int(r.get('warps',-1)))==expected,
                'paired recipe is outside the confirmed cohort')
        records.append(r)
    return records


def selection(text, manifest, expected_ops=None):
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
            and r.get("route") in ("fq", "sf", "gemv", "gemv-q4-s1", "full-bf16"),
            "invalid plan receipt",
        )
        matched_labels = {"12": "MATCHED_EXACT", "13": "MATCHED_BUCKET_PREDICTED", "14": "MATCHED_ROUTER_MINIMAX", "15": "Q8_VECTOR_MEASURED"}
        matched = r.get("policy") in matched_labels
        if matched:
            require(manifest.get("smallm_matched_policy") and r.get("activation") in ("FP16", "BF16"),
                    "unbound matched compute policy")
        if r.get("policy") == "15":
            require(manifest.get("q8_vector_policy") and r.get("q") == "8" and
                    r.get("reader") == "simt-q8-vector" and r.get("variant") in ("4", "5"),
                    "unbound Q8 vector replacement")
        if r["route"] == "full-bf16":
            require(manifest.get("prefill") and int(r.get("q",0)) in range(10,15)
                    and int(r.get("n",0))>0 and int(r.get("k",0))>0
                    and (int(r.get("rows",0))>=128 if r["op"]=="dense" else int(r.get("rows",0))>=1024)
                    and r.get("cost_scope")=="ISOLATED_COMPONENT_SUM",
                    "unbound full-BF16 prefill receipt")
        elif r["route"] == "gemv-q4-s1":
            require(r.get("q")=="12" and r.get("split")=="1", "invalid measured Q4 S1 plan")
            require(r.get("activation", "FP16") in ("FP16", "BF16"), "invalid Q4 compute type")
            if r.get("activation") == "BF16" or matched:
                receipt = manifest.get("execution_receipt", {})
                compute = receipt.get("q4_decode_compute_v2", {})
                configs = receipt.get("q4_decode_configs", {}).get(f'{r.get("n")}x{r.get("k")}', [])
                config = [int(r.get(k, -1)) for k in ("reader", "variant", "warps", "values", "columns")]
                bf16 = r.get("activation") == "BF16"
                require((not bf16 or (manifest.get("compute_contract") and "bf16" in compute.get("compute", []))) and
                        r.get("selection") == (matched_labels[r["policy"]] if matched else "INITIAL_COMPUTE") and config in configs,
                        "unbound BF16 Q4 proposal or measured relabel")
        elif r["route"] == "gemv" and r.get("reader") in ("simt-reuse", "simt-q8-vector"):
            names = ("variant", "columns", "warps", "values", "split")
            config = {name:int(r.get(name, -1)) for name in names}
            inventory = manifest.get("execution_receipt", {}).get("simt_configs", {}).get(r.get("q"), [])
            bf16 = r.get("activation") == "BF16"
            compute = manifest.get("execution_receipt", {}).get("simt_compute_v2", {})
            require((manifest.get("smallm_matched_policy") if matched else manifest.get("smallm_policy")) and
                    (matched or r.get("policy") in (("11",) if bf16 else ("9", "10"))) and
                    r.get("activation") in ("FP16", "BF16") and
                    (not bf16 or (manifest.get("compute_contract") and "bf16" in compute.get("compute", []))) and
                    any(all(c.get(k)==v for k,v in config.items()) for c in inventory),
                    "unbound small-M SIMT recipe")
        elif r["route"] in ("fq", "sf"):
            m = modules.get(r.get("build"))
            require(
                m and m["parent"]["symbol"] == r.get("parent"),
                "selected parent/build not in package",
            )
            p = m["parent"]
            compute = m["identity"].get("compute_type", "f16")
            require(r.get("activation", "FP16") == ("BF16" if compute == "bf16" else "FP16"),
                    "selected module compute type differs from caller")
            if compute == "bf16":
                require(manifest.get("compute_contract") and (matched or r.get("policy") == "11"),
                        "BF16 proposal relabeled as FP16 measurement")
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
        paired_plans=paired_selection(text,manifest),
        fallbacks=fallbacks,
        prepass=prepass,
        fully_selected=not fallbacks
        and {r["op"] for r in plans} == set(expected_ops or ("dense", "grouped")),
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
    evidence = selection(text, args.manifest | dict(modules=modules), getattr(args, "expected_ops", None))
    if os.environ.get("QUACTLIZE_KPACK_COMPUTE") == "bf16":
        for p in evidence["plans"]:
            if p["route"] == "full-bf16":
                continue  # Composition has its own BF16 provider receipt below.
            expected = "BF16" if p["op"] == "grouped" else "FP16"
            actual = p.get("activation", "FP16")
            require(actual == expected,
                    f"compute scope mismatch: tensor={p.get('tensor')} op={p['op']} route={p['route']} "
                    f"rows={p.get('rows')} expected={expected} got={actual}")
    evidence["modules"] = modules
    evidence["providers"] = provider_images(args, text, evidence["plans"])
    return evidence


def provider_images(args, text, plans):
    selected = [p for p in plans if p["route"]=="full-bf16"]
    if not selected:
        return []
    receipt = args.manifest.get("prefill", {})
    image = args.bundle/receipt.get("library", "")
    require(image.is_file() and hashlib.sha256(image.read_bytes()).hexdigest()==receipt.get("library_sha256"),
            "full-BF16 composition payload differs")
    records = {}
    for line in text.splitlines():
        if "[quactlize-prefill-image]" not in line:
            continue
        match = re.search(r'tensor=(\S+) rows=(\d+) provider=(cublas|deepgemm) image="([^"\n]+)"', line)
        require(match is not None, "invalid provider image receipt")
        tensor, rows, kind, path = match.groups()
        records[tensor, rows] = (kind, Path(path).resolve(strict=True))
    output = []
    for p in selected:
        require((p["tensor"],p["rows"]) in records, "full-BF16 plan has no provider image")
        kind, image = records[p["tensor"],p["rows"]]
        require(kind == ("cublas" if p["op"]=="dense" else "deepgemm"), "provider/operator differs")
        with image.open('rb') as stream:
            image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        record = dict(tensor=p["tensor"], rows=p["rows"], op=p["op"], provider=kind,
                      library=str(image), sha256=image_hash, kernel_execution="NOT_YET_TRACED")
        if kind == "deepgemm":
            receipt = json.loads((image.parent/f'quactlize-launch-m{p["rows"]}.json').read_text())
            require(receipt.get("schema")=="quactlize.deepgemm-launch.v1" and
                    receipt.get("shape")==list(map(int,(p["rows"],p["n"],p["k"],p["experts"]))) and
                    receipt.get("files",{}).get("kernel.so")==image_hash,
                    "DeepGEMM shape/image receipt differs")
            for name in ("kernel.cu", "kernel.args"):
                require(hashlib.sha256((image.parent/name).read_bytes()).hexdigest()==receipt["files"].get(name),
                        "DeepGEMM source/ABI changed")
            record["receipt"] = receipt
        output.append(record)
    return output


class AsysSession:
    """Collect one completed request after the same process has warmed up."""
    def __init__(self, executable, output):
        from quactlize_profile_env import tool_environment
        self.executable = executable
        self.session = "kpack-proof-" + secrets.token_hex(8)
        self.report = output / "proof.asysrep"
        self.log = output / "proof-control.log"
        self.environment = tool_environment(executable, os.environ)

    def command(self, application, environment=None):
        from quactlize_profile_env import select, tool_environment
        application_env = os.environ if environment is None else environment
        self.environment = tool_environment(self.executable, application_env)
        settings = select(application_env)
        entry = Path(__file__).with_name("quactlize_profile_env.py")
        target = [sys.executable, "-I", str(entry), json.dumps(settings, sort_keys=True), *application]
        return [str(self.executable), "launch", "--trace", "hggc", "--hggc-trace-set", "kernel-activity",
                "--sample", "none", "--python-sampling", "false", "--wait", "primary", "--kill", "sigterm", "--show-output", "true",
                "--session-new", self.session, *target]

    def control(self, action, *options, check=True):
        with self.log.open("a") as log:
            subprocess.run([str(self.executable), action, "--session", self.session, *options],
                           env=self.environment, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=check)

    def start(self):
        self.control("start", "--output", str(self.report))

    def stop(self):
        self.control("stop")

    def close(self):
        # Only this uniquely named session; never stop another user's capture.
        self.control("shutdown", check=False)


def asys_preflight(executable, output, environment=None):
    """Start the service without loading a model; retry only session creation."""
    from quactlize_profile_env import service_snapshot, tool_environment
    output.mkdir(parents=True, exist_ok=False)
    env = tool_environment(executable, os.environ if environment is None else environment)
    save(output / "services-before.json", service_snapshot(executable, env))
    attempts = []
    for attempt in range(1, 3):
        folder = output / str(attempt)
        folder.mkdir()
        profile = AsysSession(executable, folder)
        command = profile.command([sys.executable, "-I", "-c", "print('KPACK_ASYS_PROBE_READY', flush=True)"], env)
        log_path = folder / "launch.log"
        print(f"KPACK_ASYS_PREFLIGHT attempt={attempt}/2 model_loaded=0 log={log_path}", flush=True)
        timed_out = False
        rc = None
        with log_path.open("x") as log:
            try:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60).returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        text = log_path.read_text(errors="replace")
        # Cleanup is confined to the probe's unique session, including failed launches.
        with profile.log.open("a") as log:
            try:
                subprocess.run([str(executable), "shutdown", "--session", profile.session],
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
            except subprocess.TimeoutExpired:
                pass
        ready = rc == 0 and "KPACK_ASYS_PROBE_READY" in text
        creation_failure = timed_out or bool(re.search(r"session .* create timeout|create template session error", text))
        attempts.append(dict(attempt=attempt, session=profile.session, rc=rc, timeout=timed_out,
                             ready=ready, session_creation_failure=creation_failure, log=str(log_path)))
        save(output / "summary.json", dict(status="PASS" if ready else "FAIL", attempts=attempts,
                                           scope="SESSION_LAUNCH_ONLY_NOT_GPU_KERNEL_CAPTURE"))
        if ready:
            save(output / "services-after.json", service_snapshot(executable, env))
            return
        if not creation_failure:
            break
    save(output / "services-after.json", service_snapshot(executable, env))
    with (output / "environment.log").open("x") as log:
        try:
            subprocess.run([str(executable), "status", "--ppu-env"], env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=30)
        except subprocess.TimeoutExpired:
            log.write("Asys environment query exceeded 30 seconds\n")
    raise ValueError(f"Asys session preflight failed before model launch; inspect {output}; benchmark results remain valid")


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
    if getattr(args, "tensor_split", None):
        command[command.index("--split-mode") + 1] = "tensor"
        command += ["-ts", args.tensor_split]
        if arm == "reference":
            start = command.index("-ot")
            del command[start:start+2]
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
        command = profile.command(command, env)
        env = profile.environment
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
                # Readiness probes may reset while the server starts listening.
                # Retrying stays inside the process/deadline checks, not inference.
                except (URLError, TimeoutError, ConnectionError, ValueError):
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
        require("[quactlize-shard]" in text if getattr(args, "tensor_split", None) else
                "CUDA0_KPACK model buffer size" in text, "no K-pack placement")
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
    asys_preflight(args.asys, args.output / "asys-preflight")
    capture = copy.copy(args)
    capture.output = args.output / "proof-request"
    capture.output.mkdir()
    for name, value in proof_parameters(args).items():
        setattr(capture, name, value)
    arm = getattr(args, "proof_arm", "native")
    source_tokens = getattr(args, "proof_tokens", None)
    tokens = (validate_tokens(json.loads(source_tokens.read_text()), capture.prompts)
              if source_tokens else None)
    profile = AsysSession(args.asys, args.output)
    proof_arm, tokens = run_arm(capture, 0, arm, tokens, profile=profile)
    with (args.output / "proof-export.log").open("x") as f:
        subprocess.run(
            [str(args.asys), "export", "--output", str(db), str(report)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=profile.environment,
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
    if plans.get('paired_plans'):
        receipt=args.manifest.get('paired_gate_up',{})
        path=args.bundle/'libquactlize_ppu_gate_up.so'
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==receipt.get('sha256'),'paired trace library differs')
        libraries.append((path,'paired-gate-up',None))
    provider_ops = {}
    for record in plans.get("providers", []):
        if record["provider"]=="deepgemm":
            key = "provider/"+record["sha256"]
            provider_ops.setdefault(key,set()).add(record["op"])
            item = (Path(record["library"]),key,key)
            if item not in libraries:
                libraries.append(item)
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
            q4 = q4_symbol_recipe(demangled)
            reuse = simt_symbol_recipe(demangled)
            paired = paired_symbol_recipe(demangled) if label=='paired-gate-up' else None
            is_provider = build in provider_ops and re.search(r"(?i)(?:bf16|bfloat16)",demangled) and re.search(r"(?i)gemm",demangled)
            if "cutlass::device_kernel<" in demangled or q4 or reuse or paired or is_provider or re.search(
                r"kpack_q(?:8|10|11|12|13|14)::", demangled
            ):
                item = symbols.setdefault(
                    name, dict(name=demangled, libraries=[], ops=[])
                )
                item["libraries"].append(label)
                if paired:
                    ops={r['op'] for r in plans.get('paired_plans',[]) if paired_matches_plan(paired,r)}
                elif build in provider_ops:
                    ops = provider_ops[build] if is_provider else set()
                elif build:
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
                        if (r["route"] == "gemv" and q and r["q"] == q[1]) or
                           q4_symbol_matches_plan(q4, r) or
                           (reuse and measured_decode_matches_plan(measured_decode_recipe(demangled),r) and
                            r.get("reader") in ("simt-reuse", "simt-q8-vector") and reuse ==
                            (int(r["q"]),1,*[int(r[k]) for k in ("variant","columns","warps","values")],
                             int(r.get("activation")=="BF16")))
                    }
                item["ops"] = sorted(set(item["ops"]) | ops)
    total, matched = activity(db, symbols)
    require(matched, "no selected native compute kernel in PPU device trace")
    ops = {op for m in matched for op in symbols[m["mangled"]]["ops"]}
    observed_providers = {label for m in matched for label in symbols[m["mangled"]]["libraries"] if label.startswith("provider/")}
    missing_providers = [r for r in plans.get("providers",[]) if "provider/"+r["sha256"] not in observed_providers]
    # A proof covers its own short request. It is not counted as an untraced
    # performance sample or as device evidence for every ABBA parent.
    expected_ops = set(getattr(args, "expected_ops", ("dense", "grouped")))
    paired_ops={op for m in matched if 'paired-gate-up' in symbols[m['mangled']]['libraries']
                for op in symbols[m['mangled']]['ops']}
    missing_paired={r['op'] for r in plans.get('paired_plans',[])}-paired_ops
    result = dict(
        kernel_execution="PASS_SHORT_REQUEST" if ops == expected_ops and not missing_providers and not missing_paired else "PARTIAL_SHORT_REQUEST",
        paired_observed_ops=sorted(paired_ops),
        paired_missing_ops=sorted(missing_paired),
        gpu_kernel_calls=total,
        matched=matched,
        observed_ops=sorted(ops),
        missing_ops=sorted(expected_ops - ops),
        expected_ops=sorted(expected_ops),
        untraced_prefill_providers=missing_providers,
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
    p.add_argument("--tensor-split", help="TP2 split, e.g. 1,1; visible devices must be exactly two")
    p.add_argument("--jit-cache", type=Path)
    p.add_argument("--jit-helper", type=Path)
    p.add_argument("--jit-python", type=Path)
    a = p.parse_args()
    if a.tensor_split:
        require(re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*", a.tensor_split), "TP2 requires two positive split weights")
    require(a.proof_only or (a.proof_arm == "native" and a.proof_tokens is None),
            "proof arm/token overrides require --proof-only")
    a.tensor_override_pattern = (inventory_pattern(json.loads(a.tensor_inventory.read_text()))
                                 if a.tensor_inventory else PATTERN)
    a.expected_ops = (json.loads(a.tensor_inventory.read_text()).get("operators", ["dense", "grouped"])
                      if a.tensor_inventory else ["dense", "grouped"])
    require(isinstance(a.expected_ops, list) and a.expected_ops
            and len(a.expected_ops) == len(set(a.expected_ops))
            and set(a.expected_ops) <= {"dense", "grouped"}, "invalid expected operator inventory")
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
