import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import quactlize_native as native
from quactlize_native import timings, selection, summarize, PATTERN


class NativeEvidence(unittest.TestCase):
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
                  patch.object(native, "run_arm", return_value=(dict(selection=selection), None)) as run,
                  patch.object(native, "activity", return_value=(10, [dict(mangled="_Zparent")]))):
                result = native.proof(args)
            self.assertEqual(run.call_args.args[0].repeats, 1)
            self.assertEqual(run.call_args.args[0].prompts, [128])
            self.assertIsInstance(run.call_args.kwargs["profile"], native.AsysSession)
            self.assertEqual(result["kernel_execution"], "PARTIAL_SHORT_REQUEST")
            self.assertEqual(result["missing_ops"], ["dense"])
            self.assertEqual(result["selection"]["fallbacks"], fallbacks)
            self.assertEqual(json.loads((args.output / "proof.json").read_text()), result)

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
                        side_effect=ConnectionResetError(104, "connection reset"),
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

    def test_capture_starts_only_after_first_request_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(binary=Path("server"), model=Path("model"), cache=Path("cache"),
                                   context=512, batch=128, generate=8, repeats=1, prompts=[128], output=Path(temp))
            events, state = [], {}
            proc, profile = MagicMock(), MagicMock()
            proc.poll.return_value, proc.returncode = None, None
            profile.command.side_effect = lambda c: c
            profile.start.side_effect = lambda: events.append("start")
            def stop():
                events.append("stop")
                proc.poll.return_value = proc.returncode = 0
            profile.stop.side_effect = stop
            profile.close.side_effect = lambda: events.append("close")
            def launch(command, **kwargs):
                state["alias"] = command[command.index("--alias")+1]
                kwargs["stdout"].write("CUDA0_KPACK model buffer size = fixture\n")
                kwargs["stdout"].flush()
                return proc
            def response(url, key, payload=None, **kwargs):
                if url.endswith("/health"):
                    return dict(status="ok")
                if url.endswith("/props"):
                    return dict(model_alias=state["alias"], total_slots=1, default_generation_settings=dict(n_ctx=512))
                if url.endswith("/tokenize"):
                    return dict(tokens=list(range(128)))
                self.assertTrue(url.endswith("/completion"))
                events.append("request")
                return dict(tokens_evaluated=128, tokens_predicted=8, stop=True, truncated=False,
                            timings=dict(cache_n=0, prompt_n=128, predicted_n=8, prompt_ms=1., predicted_ms=1.),
                            generation_settings={k: payload[k] for k in ("n_predict", "temperature", "ignore_eos", "seed")})
            with (patch.object(native.subprocess, "Popen", side_effect=launch),
                  patch.object(native, "request", side_effect=response),
                  patch.object(native, "model_selection", return_value=dict(plans=[dict(op="grouped")]))):
                result, _ = native.run_arm(args, 0, "native", None, profile=profile)
            self.assertEqual(events, ["request", "start", "request", "stop", "close"])
            self.assertEqual([r["phase"] for r in result["records"]], ["first-use", "steady"])


if __name__ == "__main__":
    unittest.main()
