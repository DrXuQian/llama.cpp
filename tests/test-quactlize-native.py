import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import quactlize_native as native
from quactlize_native import timings, selection, summarize, PATTERN


class NativeEvidence(unittest.TestCase):
    def test_trace_tensor_inventory_uses_exact_names_including_dense_q8(self):
        names = ["blk.0.attn_q.weight", "blk.0.ffn_gate_exps.weight", "output.weight"]
        pattern = native.inventory_pattern(dict(eligible=names))
        for name in names:
            self.assertIsNotNone(re.search(pattern, name))
        for name in ("blk.0.attn_output.weight", "prefixoutput.weight", "output.weight.extra"):
            self.assertIsNone(re.search(pattern, name))
        for names in ([], ["a", "a"], [None], "output.weight"):
            with self.assertRaisesRegex(ValueError, "invalid trace tensor inventory"):
                native.inventory_pattern(dict(eligible=names))

    def test_asys_parameters_support_2048_without_hidden_128_token_clamp(self):
        self.assertEqual(native.proof_parameters(SimpleNamespace()), dict(
            context=512, batch=128, generate=8, prompts=[128], repeats=1))
        self.assertEqual(native.proof_parameters(SimpleNamespace(proof_prompt=2048, proof_generate=16)), dict(
            context=2304, batch=2048, generate=16, prompts=[2048], repeats=1))
        self.assertGreater(native.proof_parameters(SimpleNamespace(proof_prompt=2048, proof_generate=256))["context"], 2304)
        for prompt, generate in ((0, 16), (1, 16), (2048, 0), (2048, 1), (True, 16)):
            with self.assertRaisesRegex(ValueError, "invalid Asys"):
                native.proof_parameters(SimpleNamespace(proof_prompt=prompt, proof_generate=generate))

    def test_shared_tokens_are_validated_without_retokenization(self):
        tokens = {"2": [7, 8]}
        self.assertIs(native.validate_tokens(tokens, [2]), tokens)
        for value in ([], {}, {"3": [7, 8]}, {"2": [7]}, {"2": [True, 8]},
                      {"2": [-1, 8]}, {"2": [1.0, 8]}, {"2": "78"}):
            with self.subTest(tokens=value), self.assertRaisesRegex(ValueError, "input token file"):
                native.validate_tokens(value, [2])

    def test_proof_only_skips_abba_and_never_claims_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, output = root / "bundle", root / "output"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(json.dumps(dict(modules=[])))
            argv = ["quactlize_native.py", "--bundle", str(bundle), "--output", str(output),
                    "--proof-only", "--proof-prompt", "2048", "--proof-generate", "16"]
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps(dict(eligible=["blk.0.attn_q.weight", "output.weight"])))
            argv += ["--tensor-inventory", str(inventory)]
            for name in ("binary", "model", "cache", "asys", "inspector"):
                argv += ["--" + name, str(root / name)]
            result = dict(kernel_execution="PARTIAL_SHORT_REQUEST", missing_ops=["dense"])
            with (patch("sys.argv", argv), patch.object(native, "proof", return_value=result) as proof,
                  patch.object(native, "run_arm") as run):
                native.main()
            run.assert_not_called()
            proof.assert_called_once()
            self.assertEqual(proof.call_args.args[0].proof_prompt, 2048)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["order"], ["native"])
            self.assertEqual(protocol["prompts"], [2048])
            self.assertEqual(protocol["prefill_token_batch"], 2048)
            self.assertEqual(protocol["dense_scope"], "BENCHMARK_INVENTORY")
            self.assertIn("attn_q", proof.call_args.args[0].tensor_override_pattern)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "TRACE_CAPTURED")
            self.assertEqual(summary["performance"], "NOT_MEASURED")
            self.assertEqual(summary["kernel_execution"], "PARTIAL_SHORT_REQUEST")
            self.assertEqual(summary["missing_native_ops"], ["dense"])

    def module_fixture(self, root, cached):
        key = "a" * 64
        bundle, cache = root / "bundle", root / "cache"
        directory = cache / key if cached else bundle / "modules" / key
        directory.mkdir(parents=True)
        library = directory / "kernel.so"
        library.write_bytes(b"module fixture")
        record = dict(key=key, parent=dict(symbol="parent", qtype=12, route="fq-grouped"),
                      sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                      path=str(library) if cached else f"modules/{key}/kernel.so")
        args = SimpleNamespace(bundle=bundle, jit_cache=cache, jit_python=Path("/python"),
                               jit_helper=Path("/helper"), manifest=dict(
                                   modules=[] if cached else [record], jit_source_contract="b"*64))
        line = ("[quactlize-plan] tensor=w op=grouped route=fq q=12 rows=8 n=512 k=2048 "
                f"parent=parent build={key} algorithm=4 split=4 grid=0 policy=1")
        return args, record, line

    def test_selected_payload_from_package_or_jit_cache(self):
        for cached in (False, True):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as temp:
                args, record, line = self.module_fixture(Path(temp), cached)
                with patch.object(native.subprocess, "check_output", return_value=json.dumps(dict(modules=[record]))) as inspect:
                    result = native.model_selection(args, line)
                self.assertEqual(inspect.call_count, int(cached))
                self.assertEqual(result["modules"][0]["origin"], "jit-cache" if cached else "package")
                self.assertFalse(result["fully_selected"])
                if cached:
                    self.assertIn("inspect", inspect.call_args.args[0])
                    self.assertIn("b"*64, inspect.call_args.args[0])

    def test_cache_inspection_errors_do_not_admit_module(self):
        for fault in ("missing", "duplicate", "payload", "path", "parent", "compiler-failure", "bad-key"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                args, record, line = self.module_fixture(Path(temp), True)
                records = [record]
                if fault == "missing":
                    records = []
                elif fault == "duplicate":
                    records *= 2
                elif fault == "payload":
                    Path(record["path"]).write_bytes(b"changed")
                elif fault == "path":
                    record["path"] = str(Path(temp) / "outside.so")
                elif fault == "parent":
                    record["parent"]["symbol"] = "other"
                elif fault == "bad-key":
                    line = line.replace("build=", "build=../")
                with patch.object(native.subprocess, "check_output",
                                  return_value=json.dumps(dict(modules=records)),
                                  side_effect=subprocess.CalledProcessError(1, ["inspect"]) if fault == "compiler-failure" else None):
                    with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                        native.model_selection(args, line)

    def test_partial_trace_is_preserved_not_relabelled_as_full_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            args, record, line = self.module_fixture(Path(temp), False)
            args.output = Path(temp) / "results"
            args.output.mkdir()
            for name in ("asys", "proof_binary", "model", "cache", "inspector"):
                setattr(args, name, Path(temp) / name)
            fallbacks = ["output.weight: native policy miss, retain legacy K-pack FQ"]
            selection = dict(plans=[dict(op="grouped", route="fq", build=record["key"])],
                             modules=[record | dict(origin="package", path=str(args.bundle / record["path"]))],
                             fallbacks=fallbacks, fully_selected=False)
            def listing(command, **kwargs):
                if "--list-elf" in command:
                    return "Func 1: _Zparent\n"
                return "void cutlass::device_kernel<parent>()\n"
            with (patch.object(native.subprocess, "run"),
                  patch.object(native.subprocess, "check_output", side_effect=listing),
                  patch.object(native, "run_arm", return_value=(dict(selection=selection,
                      records=[dict(request_sha256="request", response=dict(content="text"))]),
                      {"128": list(range(128))})) as run,
                  patch.object(native, "activity", return_value=(10, [dict(
                      mangled="_Zparent", name="void cutlass::device_kernel<parent>()",
                      libraries=[], calls=10, total_ns=200)]))):
                result = native.proof(args)
            self.assertEqual(run.call_args.args[0].repeats, 1)
            self.assertEqual(run.call_args.args[0].prompts, [128])
            self.assertIsInstance(run.call_args.kwargs["profile"], native.AsysSession)
            self.assertEqual(result["kernel_execution"], "PARTIAL_SHORT_REQUEST")
            self.assertEqual(result["missing_ops"], ["dense"])
            self.assertEqual(result["selection"]["fallbacks"], fallbacks)
            self.assertEqual(json.loads((args.output / "proof.json").read_text()), result)
            all_times = json.loads((args.output / "kernel-times.json").read_text())
            self.assertEqual(all_times["sum_kernel_ns"], 200)
            self.assertIn("NOT_WALL_LATENCY", all_times["scope"])

    def test_reference_trace_uses_original_kernels_and_shared_input(self):
        for fault in (None, "custom-kernel", "no-native-compute"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                token_file = root / "tokens.json"
                tokens = {"128": list(range(128))}
                token_file.write_text(json.dumps(tokens))
                args = SimpleNamespace(output=root, asys=root / "asys", proof_arm="reference",
                                       proof_tokens=token_file)
                name = "void mul_mat_vec_q<(ggml_type)8, 1, false, false>()"
                if fault == "custom-kernel": name = "quactlize::runtime::moe_chain_prepare<T>()"
                if fault == "no-native-compute": name = "quantize_q8_1()"
                kernels = [dict(name=name, mangled="_kernel", calls=3, total_ns=120, libraries=[])]
                with (patch.object(native.subprocess, "run"),
                      patch.object(native.subprocess, "check_output") as inspector,
                      patch.object(native, "run_arm", return_value=(dict(selection={}, records=[
                          dict(request_sha256="shared-request", response=dict(content="answer"))]), tokens)) as run,
                      patch.object(native, "activity", return_value=(3, kernels))):
                    if fault:
                        with self.assertRaisesRegex(ValueError, "reference trace"):
                            native.proof(args)
                    else:
                        result = native.proof(args)
                        self.assertEqual(result["kernel_execution"], "REFERENCE_COMPUTE_OBSERVED")
                        self.assertEqual(result["input_tokens_sha256"], native.digest(tokens))
                        self.assertEqual(result["request_sha256"], "shared-request")
                        self.assertEqual(result["selection"], {})
                        self.assertEqual(result["timing_scope"], "PROFILER_ONLY_NOT_PERFORMANCE")
                    self.assertEqual(run.call_args.args[1:4], (0, "reference", tokens))
                    inspector.assert_not_called()

    def test_startup_failure_keeps_arm_phase_and_process_status(self):
        for returncode in (None, -6):
            with (
                self.subTest(returncode=returncode),
                tempfile.TemporaryDirectory() as temp,
            ):
                args = SimpleNamespace(
                    binary=Path("unused-server"),
                    model=Path("unused-model"),
                    cache=Path("unused-cache"),
                    context=4096,
                    batch=128,
                    output=Path(temp),
                )
                proc = MagicMock()
                proc.poll.return_value = returncode
                proc.returncode = returncode
                if returncode is None:
                    proc.wait.side_effect = lambda **kwargs: setattr(
                        proc, "returncode", -15
                    )
                with (
                    patch.object(native.subprocess, "Popen", return_value=proc),
                    patch.object(
                        native,
                        "request",
                        side_effect=PermissionError(13, "permission denied"),
                    ) as request,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "1-native phase=startup-health"
                    ):
                        native.run_arm(args, 1, "native", None)
                failure = json.loads(
                    (args.output / "1-native.failure.json").read_text()
                )
                self.assertEqual(failure["returncode_before_cleanup"], returncode)
                self.assertEqual(failure["log"], str(args.output / "1-native.log"))
                self.assertEqual(request.call_count, int(returncode is None))
                self.assertEqual(proc.terminate.call_count, int(returncode is None))
                proc.kill.assert_not_called()

    def test_health_reset_does_not_hide_process_exit_or_startup_deadline(self):
        for stopped in (True, False):
            with self.subTest(stopped=stopped), tempfile.TemporaryDirectory() as temp:
                args = SimpleNamespace(binary=Path("server"), model=Path("model"), cache=Path("cache"),
                                       context=512, batch=128, output=Path(temp))
                proc = MagicMock()
                if stopped:
                    statuses = iter((None, -6))
                    proc.poll.side_effect = lambda: next(statuses, -6)
                    proc.returncode = -6
                else:
                    proc.poll.return_value = proc.returncode = None
                    proc.wait.side_effect = lambda **kwargs: setattr(proc, "returncode", -15)
                with (patch.object(native.subprocess, "Popen", return_value=proc),
                      patch.object(native, "request", side_effect=ConnectionResetError(104, "connection reset")) as request,
                      patch.object(native.time, "sleep") as sleep,
                      patch.object(native.time, "monotonic", side_effect=[0, 0, 0, 901])):
                    reason = "server exited" if stopped else "startup exceeded 900 seconds"
                    with self.assertRaisesRegex(ValueError, reason):
                        native.run_arm(args, 0, "native", None)
                request.assert_called_once()
                sleep.assert_called_once_with(1)
                failure = json.loads((args.output / "0-native.failure.json").read_text())
                self.assertEqual(failure["phase"], "startup-health")
                self.assertEqual(failure["returncode_before_cleanup"], -6 if stopped else None)
                proc.kill.assert_not_called()

    def test_completion_reset_is_not_retried_as_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(binary=Path("server"), model=Path("model"), cache=Path("cache"),
                                   context=512, batch=128, generate=8, repeats=1, prompts=[128], output=Path(temp))
            proc = MagicMock()
            proc.poll.return_value = proc.returncode = None
            proc.wait.side_effect = lambda **kwargs: setattr(proc, "returncode", -15)
            props = dict(model_alias="native-fixture", total_slots=1, default_generation_settings=dict(n_ctx=512))
            with (patch.object(native.subprocess, "Popen", return_value=proc),
                  patch.object(native.secrets, "token_hex", return_value="fixture"),
                  patch.object(native.time, "sleep") as sleep,
                  patch.object(native, "request", side_effect=[dict(status="ok"), props,
                               ConnectionResetError(104, "completion reset")]) as request):
                with self.assertRaisesRegex(ValueError, "phase=completion-prompt-128-repeat-0"):
                    native.run_arm(args, 0, "native", {"128": list(range(128))})
            self.assertEqual(request.call_count, 3)
            sleep.assert_not_called()
            proc.terminate.assert_called_once_with()
            failure = json.loads((args.output / "0-native.failure.json").read_text())
            self.assertIn("completion reset", failure["error"])
            self.assertFalse((args.output / "0-native.selection.json").exists())

    def test_tensor_override_matches_complete_weight_names(self):
        positive = ["output.weight"] + [
            f"blk.{i}.ffn_{part}_exps.weight"
            for i in range(40)
            for part in ("gate", "up", "down", "gate_up")
        ]
        negative = [
            f"blk.{i}.{part}.weight"
            for i in range(40)
            for part in ("attn_output", "attn_q", "attn_k", "attn_v", "ffn_up_shexp")
        ] + [
            "token_embd.weight",
            "output.bias",
            "output.weight.extra",
            "prefixoutput.weight",
            "blk.3.ffn_down_exps.bias",
            "prefix.blk.3.ffn_down_exps.weight",
        ]
        names = positive + negative
        want = [True] * len(positive) + [False] * len(negative)
        self.assertEqual([bool(re.search(PATTERN, name)) for name in names], want)
        # Exercise the same C++ regex_search semantics as the loader.
        source = r"""
#include <iostream>
#include <regex>
#include <string>
int main(int argc, char ** argv) {
    const std::regex pattern(argv[1]);
    for (int i = 2; i < argc; ++i) {
        std::cout << std::regex_search(std::string(argv[i]), pattern) << '\n';
    }
}
"""
        with tempfile.TemporaryDirectory() as temp:
            binary = str(Path(temp) / "regex-search")
            subprocess.run(
                ["c++", "-std=c++17", "-x", "c++", "-", "-o", binary],
                input=source,
                text=True,
                check=True,
                capture_output=True,
            )
            got = subprocess.check_output([binary, PATTERN, *names], text=True)
            self.assertEqual(got.splitlines(), [str(int(value)) for value in want])
            legacy = subprocess.check_output(
                [binary, r"(ffn_.*_exps|output\.weight)", "blk.3.attn_output.weight"],
                text=True,
            )
            self.assertEqual(legacy, "1\n")

    def test_timers(self):
        p = dict(
            prompt=[1] * 128, n_predict=16, temperature=0.0, ignore_eos=True, seed=1
        )
        r = dict(
            tokens_evaluated=128,
            tokens_predicted=16,
            stop=True,
            truncated=False,
            timings=dict(
                prompt_n=128,
                predicted_n=16,
                prompt_ms=128.0,
                predicted_ms=32.0,
                cache_n=0,
            ),
            generation_settings={k: v for k, v in p.items() if k != "prompt"},
        )
        self.assertEqual(timings(r, p)["decode_us_per_token"], 2000.0)
        for key, value in (
            ("cache_n", 1),
            ("predicted_ms", float("nan")),
            ("prompt_n", 127),
        ):
            bad = copy.deepcopy(r)
            bad["timings"][key] = value
            with self.assertRaises(ValueError):
                timings(bad, p)

    def test_selection_binding_and_fallback(self):
        manifest = dict(
            modules=[
                dict(
                    key="a" * 64, parent=dict(symbol="p", qtype=12, route="sf-grouped")
                )
            ]
        )
        line = (
            "[quactlize-plan] tensor=w op=grouped route=sf q=12 rows=1024 n=512 k=2048 parent=p build="
            + "a" * 64
            + " algorithm=0 split=1 grid=0 policy=4"
        )
        result = selection(line, manifest)
        self.assertFalse(result["fully_selected"])
        line += "\n[quactlize-plan] tensor=output.weight op=dense route=gemv q=14 rows=1 columns=32 warps=4 split=1 selection=MEASURED_GEMV_POOL"
        self.assertTrue(selection(line, manifest)["fully_selected"])
        with self.assertRaises(ValueError):
            selection(line.replace("parent=p", "parent=other"), manifest)
        line += (
            "\n[quactlize] output.weight: native policy miss, retain legacy K-pack FQ"
        )
        self.assertFalse(selection(line, manifest)["fully_selected"])
        self.assertIn("output", PATTERN)

    def test_abba(self):
        arms = []
        for arm in ("reference", "native", "native", "reference"):
            rows = [
                dict(
                    prompt=128,
                    repeat=i,
                    phase="steady",
                    request_sha256="same",
                    timings=dict(prefill_us_per_token=1.0, decode_us_per_token=2.0),
                    response=dict(content="same"),
                )
                for i in (1, 2)
            ]
            rows.append(dict(prompt=128, phase="first-use", request_sha256="warmup",
                             timings=dict(prefill_us_per_token=1e9, decode_us_per_token=1e9)))
            arms.append(dict(arm=arm, records=rows))
        self.assertEqual(
            summarize(arms, [128], 2)[0]["decode_us_per_token"]["delta_pct"], 0.0
        )
        arms[1]["records"][0]["request_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            summarize(arms, [128], 2)

    def test_missing_or_duplicate_first_use_cannot_be_steady_evidence(self):
        for count in (0, 2):
            arms = []
            for arm in ("reference", "native", "native", "reference"):
                rows = [dict(prompt=128, phase="first-use") for _ in range(count)]
                arms.append(dict(arm=arm, records=rows))
            with self.assertRaisesRegex(ValueError, "excluded first-use"):
                summarize(arms, [128], 2)

    def test_asys_session_uses_explicit_start_stop_not_a_startup_delay(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = native.AsysSession(Path("/asys"), Path(temp))
            command = profile.command(["server", "--model", "model.gguf"])
            self.assertEqual(command[:2], ["/asys", "launch"])
            self.assertNotIn("--delay", command)
            self.assertIn(profile.session, command)
            with patch.object(native.subprocess, "run") as run:
                profile.start()
                profile.stop()
                profile.close()
            self.assertEqual([c.args[0][1] for c in run.call_args_list], ["start", "stop", "shutdown"])
            self.assertTrue(all(c.args[0][3] == profile.session for c in run.call_args_list))

    def test_profile_target_replaces_stale_route_environment_before_exec(self):
        from quactlize_profile_env import select
        with tempfile.TemporaryDirectory() as temp:
            profile = native.AsysSession(Path("/asys"), Path(temp))
            stale = dict(os.environ, QUACTLIZE_KPACK_EXECUTION="stale-reference",
                         QUACTLIZE_KPACK_PAIR_WEIGHTS="0", LLAMA_ARG_MODEL="wrong-model",
                         GGML_CUDA_DISABLE_GRAPHS="1", NSIGHT_TEST_INJECTION="preserved")
            app = [sys.executable, "-I", "-c",
                   "import json,os; print(json.dumps(dict(os.environ),sort_keys=True))"]
            for arm in ("reference", "native"):
                settings = dict(PATH=os.environ["PATH"], CUDA_VISIBLE_DEVICES="0")
                if arm == "native":
                    settings.update(QUACTLIZE_KPACK_EXECUTION="/bundle with spaces,$literal",
                                    QUACTLIZE_KPACK_JIT_HELPER="/repo/helper.py",
                                    QUACTLIZE_KPACK_PAIR_WEIGHTS="1")
                command = profile.command(app, settings)
                # Model a daemon launching with the PREVIOUS arm's environment.
                target = command[command.index(sys.executable):]
                run = subprocess.run(target, env=stale, text=True, capture_output=True, check=True)
                receipt, actual = run.stdout.splitlines()
                self.assertEqual(json.loads(receipt.removeprefix("KPACK_PROFILE_ENV ")), settings)
                environment = json.loads(actual)
                self.assertEqual(select(environment), settings)
                self.assertEqual(environment["NSIGHT_TEST_INJECTION"], "preserved")
                self.assertEqual(stale["QUACTLIZE_KPACK_EXECUTION"], "stale-reference")

    def captured_exit(self, exit_status, arm="native", shared_tokens=None, health_errors=()):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(binary=Path("server"), model=Path("model"), cache=Path("cache"),
                                   context=512, batch=128, generate=8, repeats=1, prompts=[128], output=Path(temp))
            events, state = [], {}
            proc, profile = MagicMock(), MagicMock()
            proc.poll.return_value, proc.returncode = None, None
            profile.command.side_effect = lambda c, env: c
            profile.start.side_effect = lambda: events.append("start")
            def stop():
                events.append("stop")
            profile.stop.side_effect = stop
            profile.close.side_effect = lambda: events.append("close")
            def terminate():
                state["second_signal"] = True
            proc.terminate.side_effect = terminate
            def wait(timeout=None):
                events.append("wait")
                if not proc.kill.called:
                    self.assertEqual(timeout, 60)
                    if exit_status == "timeout":
                        raise subprocess.TimeoutExpired("asys launch", timeout)
                proc.poll.return_value = proc.returncode = (
                    -9 if proc.kill.called else 1 if state.get("second_signal") else exit_status)
                return proc.returncode
            proc.wait.side_effect = wait
            def launch(command, **kwargs):
                state["alias"] = command[command.index("--alias")+1]
                for key in ("GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION"):
                    self.assertNotIn(key, kwargs["env"])
                if arm == "reference":
                    self.assertFalse(any(k.startswith("QUACTLIZE_KPACK_") for k in kwargs["env"]))
                    self.assertTrue(command[command.index("-ot")+1].endswith("=CUDA0"))
                    self.assertNotIn("--kpack-cache", command)
                else:
                    kwargs["stdout"].write("CUDA0_KPACK model buffer size = fixture\n")
                kwargs["stdout"].flush()
                return proc
            pending_health = iter(health_errors)
            def response(url, key, payload=None, **kwargs):
                if url.endswith("/health"):
                    error = next(pending_health, None)
                    if error is not None:
                        raise error
                    return dict(status="ok")
                if url.endswith("/props"):
                    return dict(model_alias=state["alias"], total_slots=1, default_generation_settings=dict(n_ctx=512))
                if url.endswith("/tokenize"):
                    self.assertIsNone(shared_tokens)
                    return dict(tokens=list(range(128)))
                self.assertTrue(url.endswith("/completion"))
                if shared_tokens: self.assertEqual(payload["prompt"], shared_tokens["128"])
                events.append("request")
                return dict(tokens_evaluated=128, tokens_predicted=8, stop=True, truncated=False,
                            timings=dict(cache_n=0, prompt_n=128, predicted_n=8, prompt_ms=1., predicted_ms=1.),
                            generation_settings={k: payload[k] for k in ("n_predict", "temperature", "ignore_eos", "seed")})
            with (patch.dict(native.os.environ, {"QUACTLIZE_KPACK_EXECUTION": "/fixture",
                  "QUACTLIZE_KPACK_ROUTE": "gemv", "QUACTLIZE_KPACK_PAIR_WEIGHTS": "1",
                  "QUACTLIZE_KPACK_JIT_HELPER": "/fixture-helper", "GGML_CUDA_DISABLE_GRAPHS": "1",
                  "GGML_CUDA_DISABLE_FUSION": "1"}),
                  patch.object(native.subprocess, "Popen", side_effect=launch),
                  patch.object(native, "request", side_effect=response),
                  patch.object(native.time, "sleep") as sleep,
                  patch.object(native, "model_selection", return_value=dict(plans=[dict(op="grouped")])) as select):
                if exit_status == 0:
                    result, _ = native.run_arm(args, 0, arm, shared_tokens, profile=profile)
                    self.assertEqual([r["phase"] for r in result["records"]], ["first-use", "steady"])
                    if arm == "reference": select.assert_not_called()
                else:
                    message = "shutdown exceeded 60 seconds" if exit_status == "timeout" else "0-native failed rc=1"
                    with self.assertRaisesRegex(ValueError, message):
                        native.run_arm(args, 0, arm, shared_tokens, profile=profile)
            self.assertEqual(sleep.call_count, len(health_errors))
            self.assertEqual(events, ["request", "start", "request", "stop", "close", "wait"] +
                             (["wait"] if exit_status == "timeout" else []))
            proc.terminate.assert_not_called()
            profile.close.assert_called_once_with()
            if exit_status == "timeout":
                proc.kill.assert_called_once_with()
                self.assertEqual(proc.wait.call_count, 2)
            else:
                proc.kill.assert_not_called()
                proc.wait.assert_called_once_with(timeout=60)
            self.assertEqual(json.loads((args.output / f"0-{arm}.process.json").read_text()),
                             dict(returncode=-9 if exit_status == "timeout" else exit_status))
            self.assertEqual(json.loads((args.output / "input-tokens.json").read_text()),
                             shared_tokens or {"128": list(range(128))})

    def test_capture_starts_only_after_first_request_completed(self):
        self.captured_exit(0)

    def test_transient_health_reset_recovers_before_warmup_and_capture(self):
        from http.client import RemoteDisconnected
        self.captured_exit(0, health_errors=(ConnectionResetError(104, "connection reset"),
                                            ConnectionRefusedError(111, "not listening"),
                                            RemoteDisconnected("not ready")))

    def test_reference_restores_native_fusions_and_clears_kpack_environment(self):
        self.captured_exit(0, "reference", {"128": list(range(900, 1028))})

    def test_profile_shutdown_does_not_accept_real_nonzero_exit(self):
        self.captured_exit(1)

    def test_profile_shutdown_timeout_preserves_status_and_stays_failure(self):
        self.captured_exit("timeout")


if __name__ == "__main__":
    unittest.main()
