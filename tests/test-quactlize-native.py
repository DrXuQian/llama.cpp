import copy
import unittest
from quactlize_native import timings, selection, summarize, PATTERN


class NativeEvidence(unittest.TestCase):
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
