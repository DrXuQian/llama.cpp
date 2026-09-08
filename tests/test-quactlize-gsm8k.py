#!/usr/bin/env python3
"""Host protocol/scorer checks; the fake server is not a model oracle."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import quactlize_gsm8k as gsm


def response(payload, content="Some reasoning.\n#### 42", **changes):
    row = {"content": content, "stop": True, "id_slot": 0, "tokens_predicted": 8,
           "tokens_evaluated": len(payload["prompt"]), "timings": {"cache_n": 0},
           "generation_settings": {k: v for k, v in payload.items() if k != "prompt"},
           "stop_type": "eos", "truncated": False}
    row.update(changes)
    return row


def fake_server(argv):
    """Exercise the actual subprocess/HTTP lifecycle without loading any model."""
    assert not any(key.startswith("LLAMA_ARG_") for key in os.environ)
    if "--help" in argv:
        return
    value = lambda name: argv[argv.index(name) + 1]
    kpack = "--kpack-cache" in argv
    print("CUDA0_KPACK model buffer size = 20.00 MiB" if kpack else "CUDA0 model buffer size = 20.00 MiB", flush=True)
    if kpack:
        print("[kpack-cache] cache_uploads=1 resident_misses=0", flush=True)

    class Handler(BaseHTTPRequestHandler):
        completions = 0

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def respond(self, data):
            if self.headers.get("Authorization") != "Bearer " + value("--api-key"):
                self.send_error(401)
                return
            if self.path == "/health":
                result = {"status": "ok"}
            elif self.path == "/props":
                result = {"model_alias": value("--alias"), "total_slots": int(value("-np")),
                          "default_generation_settings": {"n_ctx": int(value("-c"))},
                          "chat_template": "mock template", "model_path": value("-m"), "build_info": "host-test"}
            elif self.path == "/apply-template":
                assert list(data) == ["messages", "chat_template_kwargs"]
                assert len(data["messages"]) == 1 and data["chat_template_kwargs"]["enable_thinking"] is False
                assert "PRIVATE_GOLD_SOLUTION" not in json.dumps(data)
                result = {"prompt": "USER: " + data["messages"][0]["content"] + "\nASSISTANT:"}
            elif self.path == "/tokenize":
                assert data["add_special"] and data["parse_special"]
                result = {"tokens": list(data["content"].encode())}
            elif self.path == "/completion":
                Handler.completions += 1
                if kpack and Handler.completions == 2 and os.environ.get("FAKE_KPACK_FAILURE") == "1":
                    self.send_error(500, "injected failure")
                    return
                assert all(type(t) is int for t in data["prompt"])
                assert data["cache_prompt"] is False and data["stream"] is False and data["id_slot"] == 0
                result = response(data)
            else:
                self.send_error(404)
                return
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            pass

    HTTPServer(("127.0.0.1", int(value("--port"))), Handler).serve_forever()


class AnswerProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = gsm.completion_payload([1, 2, 3], 9, 1024)
        self.manifest = {"tensors": [{"name": "weight", "route_class": "grouped"}]}

    def test_exact_numeric_normalization(self):
        for text, want in (("42", "42"), ("1,234.00", "1234"), ("-4.50", "-4.5"),
                           ("+.25", "0.25"), ("-0.00", "0"), ("100", "100")):
            self.assertEqual(gsm.number(text), want)

    def test_no_broken_commas_units_nan_fractions_or_exponents(self):
        for text in ("NaN", "inf", "1,23", "42 dollars", "$42", "4/2", "4e2", "**42**", "42."):
            with self.subTest(text=text):
                self.assertIsNone(gsm.number(text))

    def test_only_terminal_final_marker(self):
        self.assertEqual(gsm.final_answer("1+2=3\n#### 42\n\n"), "42")
        for text in ("The answer is 42", "42", "#### 42\nActually 40", "#### 42 and 43", "####", "#### 42\n</think>"):
            self.assertIsNone(gsm.final_answer(text))

    def test_correct_wrong_parse_and_cap(self):
        self.assertTrue(gsm.score(response(self.payload), self.payload, "42")["correct"])
        self.assertFalse(gsm.score(response(self.payload), self.payload, "41")["correct"])
        self.assertTrue(gsm.score(response(self.payload, "reasoning 42"), self.payload, "42")["parse_failed"])
        for change in ({"stop_type": "limit"}, {"truncated": True}):
            result = gsm.score(response(self.payload, **change), self.payload, "42")
            self.assertTrue(result["truncated"])
            self.assertFalse(result["correct"])

    def test_response_negatives(self):
        for change in ({"stop": False}, {"id_slot": 1}, {"tokens_evaluated": 4}, {"tokens_predicted": 0},
                       {"tokens_predicted": 1025}, {"truncated": None}, {"stop_type": "none"},
                       {"content": None}, {"timings": {"cache_n": 3}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                gsm.score(response(self.payload, **change), self.payload, "42")

    def test_effective_generation_settings(self):
        for key in ("seed", "n_predict", "temperature", "top_k", "repeat_penalty", "ignore_eos"):
            row = response(self.payload)
            row["generation_settings"][key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "setting differs"):
                gsm.score(row, self.payload, "42")

    def dataset(self, suffix=".json"):
        rows = [{"question": f"Question number {i}?", "answer": "PRIVATE_GOLD_SOLUTION\n#### 42"} for i in range(130)]
        path = self.root / ("test" + suffix)
        if suffix == ".jsonl":
            path.write_text("\n".join(json.dumps(row) for row in rows))
        elif suffix == ".parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=8)
        else:
            gsm.save(path, rows)
        return path

    def test_seeded_unique_question_only(self):
        source = self.dataset()
        cases = gsm.prepare(source, 128, 12)
        self.assertEqual(cases, gsm.prepare(source, 128, 12))
        self.assertNotEqual(cases, gsm.prepare(source, 128, 13))
        self.assertEqual(len({row["id"] for row in cases}), 128)
        self.assertTrue(all("PRIVATE_GOLD_SOLUTION" not in row["prompt"] for row in cases))
        for size in (0, 131):
            with self.assertRaises(ValueError):
                gsm.prepare(source, size, 12)
        bad = self.root / "bad.json"
        gsm.save(bad, [{"question": "question", "answer": "no final marker"}])
        with self.assertRaisesRegex(ValueError, "gold"):
            gsm.prepare(bad, 1, 12)

    def test_jsonl_full_dataset(self):
        self.assertEqual(gsm.prepare(self.dataset(".jsonl"), 130, 7), gsm.prepare(self.dataset(), 130, 7))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet reader unavailable")
    def test_parquet_row_groups_full_sample(self):
        self.assertEqual(gsm.prepare(self.dataset(".parquet"), 128, 7), gsm.prepare(self.dataset(), 128, 7))

    def test_route_evidence_does_not_claim_trace(self):
        result = gsm.route_receipt("CUDA0_KPACK model buffer size = 20.00 MiB\ncache_uploads=1 resident_misses=0", self.manifest, "kpack")
        self.assertEqual(result["kernel_execution"], "NOT_COLLECTED")
        gsm.route_receipt("CUDA0 model buffer size = 20.00 MiB", self.manifest, "reference")

    def test_route_and_cache_negatives(self):
        good = "CUDA0_KPACK model buffer size = 20.00 MiB\ncache_uploads=1 resident_misses=0"
        for text in ("", good + "\nGPU pack queued", good.replace("misses=0", "misses=1"), good + "\nCUDA error:"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                gsm.route_receipt(text, self.manifest, "kpack")
        with self.assertRaises(ValueError):
            gsm.route_receipt(good, self.manifest, "reference")

    def write_pair(self):
        cases = [{"id": i, "gold": "42"} for i in range(3)]
        for arm in ("reference", "kpack"):
            rows = []
            for i, case in enumerate(cases):
                payload = gsm.completion_payload([1, 2, 3], 9 + i, 1024)
                content = "#### 42" if i == (0 if arm == "reference" else 1) else "no final answer"
                rows.append({"id": i, "input_tokens": payload["prompt"], "request_sha256": gsm.digest(payload),
                             "response": response(payload, content)})
            (self.root / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return cases

    def test_paired_summary_keeps_failed_parses(self):
        result = gsm.summarize(self.root, self.write_pair(), 9, 1024)
        self.assertEqual(result["arms"]["kpack"]["questions"], 3)
        self.assertEqual(result["arms"]["kpack"]["correct"], 1)
        self.assertEqual(result["arms"]["kpack"]["parse_failed"], 2)
        self.assertEqual(result["reference_correct_kpack_wrong"], [0])
        self.assertEqual(result["reference_wrong_kpack_correct"], [1])
        self.assertEqual(result["accuracy_admission"], "PENDING_REVIEW")

    def test_summary_missing_duplicate_and_mismatched_inputs(self):
        for mode in ("missing", "duplicate", "different"):
            cases = self.write_pair()
            path = self.root / "kpack.jsonl"
            rows = gsm.read_rows(path)
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[1]["id"] = 0
            else:
                payload = gsm.completion_payload([3, 2, 1], 9, 1024)
                rows[0].update(input_tokens=payload["prompt"], request_sha256=gsm.digest(payload))
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                gsm.summarize(self.root, cases, 9, 1024)

    def fake_binary(self, binary):
        binary.write_text(f"#!{sys.executable}\nimport runpy, sys\n"
                          f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
                          f"runpy.run_path({str(Path(__file__).resolve())!r})['fake_server'](sys.argv[1:])\n")
        binary.chmod(0o755)

    def test_two_real_processes_with_local_fake_http(self):
        binary = self.root / "server"
        self.fake_binary(binary)
        args = argparse.Namespace(output=self.root, binary=binary, model=self.root / "model",
                                  cache=self.root, context=4096, batch=128, max_tokens=1024, seed=9)
        cases = gsm.prepare(self.dataset(), 2, 9)
        for arm in ("reference", "kpack"):
            receipt = gsm.run_arm(args, arm, cases, self.manifest)
            self.assertEqual(receipt["placement"], "PASS")
            command = json.loads((self.root / f"{arm}.command.json").read_text())
            self.assertEqual(command[command.index("-np") + 1], "1")
            self.assertEqual(command[-1], "<ephemeral-local-key>")
        result = gsm.summarize(self.root, cases, args.seed, args.max_tokens)
        self.assertEqual(result["arms"]["reference"]["correct"], 2)
        self.assertEqual(result["arms"]["kpack"]["correct"], 2)

    def test_full_wrapper_and_partial_failure_archive(self):
        test_dir = self.root / "tests"
        test_dir.mkdir()
        for name in ("quactlize_numerical.py", "quactlize_gsm8k.py", "run-quactlize-gsm8k.sh"):
            shutil.copyfile(Path(__file__).with_name(name), test_dir / name)
        build = self.root / "build"
        (build / "bin").mkdir(parents=True)
        (build / "CMakeCache.txt").write_text("GGML_USE_PPU:BOOL=ON\nGGML_NCP_QUACTLIZE:BOOL=ON\n"
                                            "GGML_NCP_FA:BOOL=OFF\nGGML_NCP_MOE:BOOL=OFF\nGGML_NCP_GDN:BOOL=OFF\n")
        self.fake_binary(build / "bin/llama-server")
        for lib in ("libggml-cuda.so", "libllama.so"):
            (build / "bin" / lib).write_text("host fixture")
        sdk = self.root / "sdk"
        sdk.mkdir()
        (sdk / "envsetup.sh").write_text("# host fixture\n")
        bundle = self.root / "bundle"
        bundle.mkdir()
        gsm.save(bundle / "manifest.json", {})
        for lib in [f"libquactlize_ppu_fmt{fmt}.so" for fmt in range(5)] + ["pack.so"]:
            (bundle / lib).write_text("host fixture")
        gsm.save(self.root / "manifest.json", self.manifest)
        (self.root / "model.gguf").write_text("host fixture, not a model")
        fake_path = self.root / "bin"
        fake_path.mkdir()
        for name, body in (("git", 'if [ "$1" = rev-parse ]; then printf "fixture-source\\n"; fi'),
                           ("cmake", 'printf "fake-cmake %s\\n" "$*"')):
            file = fake_path / name
            file.write_text("#!/bin/sh\n" + body + "\n")
            file.chmod(0o755)
        env = {**os.environ, "MODEL": str(self.root / "model.gguf"), "PPU_SDK": str(sdk),
               "QUACTLIZE_PPU_BUNDLE": str(bundle), "QUACTLIZE_PPU_PACK_LIBRARY": str(bundle / "pack.so"),
               "CACHE_DIR": str(self.root), "GSM8K_FILE": str(self.dataset()), "BUILD_DIR": str(build),
               "RESULT_ROOT": str(self.root), "JOBS": "192", "GSM8K_CASES": "2", "CUDA_VISIBLE_DEVICES": "0",
               "PATH": str(fake_path) + os.pathsep + os.environ["PATH"], "LLAMA_ARG_PARALLEL": "32"}
        for failure in (False, True):
            with self.subTest(failure=failure):
                result = subprocess.run(["bash", "tests/run-quactlize-gsm8k.sh"], cwd=self.root,
                                        env={**env, "FAKE_KPACK_FAILURE": str(int(failure))},
                                        capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 1 if failure else 0, result.stdout + result.stderr)
                archive = Path(next(line[8:] for line in result.stdout.splitlines() if line.startswith("results=")))
                self.assertTrue(archive.is_file())
                root = Path(str(archive).removesuffix(".results.tgz")) / "results"
                self.assertEqual(len(gsm.read_rows(root / "reference.jsonl")), 2)
                self.assertEqual(len(gsm.read_rows(root / "kpack.jsonl")), 1 if failure else 2)
                self.assertEqual((root / "summary.json").exists(), not failure)
                self.assertIn("--target llama-server -j 192", (root / "build.log").read_text())

    def test_shell_syntax_and_help(self):
        script = Path(__file__).with_name("run-quactlize-gsm8k.sh")
        subprocess.run(["bash", "-n", str(script)], check=True)
        result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=True)
        self.assertIn("Business request batch is always 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
