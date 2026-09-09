import copy
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
            arms.append(dict(arm=arm, records=rows))
        self.assertEqual(
            summarize(arms, [128], 2)[0]["decode_us_per_token"]["delta_pct"], 0.0
        )
        arms[1]["records"][0]["request_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            summarize(arms, [128], 2)


if __name__ == "__main__":
    unittest.main()
