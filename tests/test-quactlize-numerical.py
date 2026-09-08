#!/usr/bin/env python3
"""Host checks for the numerical runner's evidence parser, not a model oracle."""

import copy
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

from quactlize_numerical import activity, analyze, gsm8k_corpus, logprobs


class NumericalEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "trace.sqlite"
        self.log = self.root / "model.log"
        self.index = {"kernels": {"_kernel": {"name": "exact grouped GEMM", "libraries": ["libquactlize_ppu_fmt0.so"]}}}
        self.manifest = {"tensors": [{"name": "blk.0.ffn_up_exps.weight", "route_class": "grouped", "ggml_type": 12}]}
        self.text = (
            'cache_uploads=1 resident_misses=0\n'
            'kl_divergence: computing over 2 chunks, n_ctx=256, batch_size=1, n_seq=1\n'
            '[ncp-route] MUL_MAT_ID blk.0.ffn_up_exps.weight T=1 E=256 used=8 -> so-quactlize-kpack\n'
            'Mean    KLD: 0.000001\nMaximum KLD: 0.000010\n'
            'Mean PPL(Q)/PPL(base) : 1.000000\nSame top p: 100.000\n'
        )
        con = sqlite3.connect(self.db)
        con.executescript('''
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES(1, '_kernel'), (2, 'exact grouped GEMM');
            CREATE TABLE HGPTI_ACTIVITY_KIND_KERNEL(start INT, end INT, mangledName INT, demangledName INT);
            WITH RECURSIVE seq(x) AS (SELECT 0 UNION ALL SELECT x+1 FROM seq WHERE x<511)
            INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL SELECT x*20, x*20+10, 1, 2 FROM seq;
        ''')
        con.close()

    def check(self, phase="cache-reference", text=None, manifest=None):
        self.log.write_text(self.text if text is None else text)
        return analyze(self.log, self.db, self.index, manifest or self.manifest, 1, phase)

    def edit_db(self, sql):
        with sqlite3.connect(self.db) as con:
            con.executescript(sql)

    def test_positive(self):
        result = self.check()
        self.assertEqual(result["quactlize_grouped_gemm_calls"], 512)
        self.assertEqual(result["accuracy_admission"], "PENDING_REVIEW")

    def test_route_text_alone_is_not_execution(self):
        self.edit_db("UPDATE StringIds SET value='pack_or_gather' WHERE id IN (1, 2)")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.check()

    def test_empty_trace(self):
        self.edit_db("DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL")
        with self.assertRaisesRegex(ValueError, "empty GPU"):
            self.check()

    def test_api_trace_is_not_kernel_trace(self):
        self.edit_db("ALTER TABLE HGPTI_ACTIVITY_KIND_KERNEL RENAME TO HGPTI_ACTIVITY_KIND_RUNTIME")
        with self.assertRaisesRegex(ValueError, "device-kernel activity"):
            self.check()

    def test_incomplete_kernel(self):
        self.edit_db("UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET end=start")
        with self.assertRaisesRegex(ValueError, "duration"):
            self.check()

    def test_warmup_sized_trace(self):
        self.edit_db("DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL WHERE start>0")
        with self.assertRaisesRegex(ValueError, "incomplete evaluation kernel trace"):
            self.check()

    def test_demangled_fallback(self):
        self.edit_db("UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET mangledName=NULL")
        self.assertEqual(self.check()["quactlize_grouped_gemm_calls"], 512)

    def test_reference_negative_control(self):
        text = self.text.replace('so-quactlize-kpack', 'GENERIC')
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.check("reference-self", text)
        self.edit_db("UPDATE StringIds SET value='ordinary_gpu_gemm' WHERE id IN (1, 2)")
        self.assertEqual(self.check("reference-self", text)["quactlize_grouped_gemm_calls"], 0)

    def test_warmup_only_route(self):
        lines = self.text.splitlines(keepends=True)
        text = lines[2] + lines[0] + lines[1] + ''.join(lines[3:])
        with self.assertRaisesRegex(ValueError, "route coverage"):
            self.check(text=text)

    def test_partial_route_coverage(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['tensors'].append(dict(manifest['tensors'][0], name='blk.1.ffn_up_exps.weight'))
        with self.assertRaisesRegex(ValueError, "missing="):
            self.check(manifest=manifest)

    def test_wrong_batch(self):
        with self.assertRaisesRegex(ValueError, "route coverage"):
            self.check(text=self.text.replace('T=1 E=', 'T=128 E='))

    def test_nonfinite_and_missing_metrics(self):
        for text in (self.text.replace('0.000001', 'nan'), self.text.replace('Mean    KLD:', 'not KLD:')):
            with self.assertRaises(ValueError):
                self.check(text=text)

    def test_cache_miss(self):
        with self.assertRaisesRegex(ValueError, "full cache hit"):
            self.check(text=self.text.replace('resident_misses=0', 'resident_misses=1'))

    def test_runtime_failure(self):
        with self.assertRaisesRegex(ValueError, "runtime error"):
            self.check(text=self.text + 'CUDA error: invalid kernel image\n')

    def test_logprobs_coverage(self):
        path = self.root / 'logprobs'
        payload = b'_logits_' + struct.pack('<iii', 256, 8, 2) + bytes(512 * 4) + bytes(254 * 12 * 2)
        path.write_bytes(payload)
        self.assertEqual(logprobs(path)['scored_tokens'], 254)
        path.write_bytes(payload[:-2])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            logprobs(path)

    def test_gsm8k_jsonl_sample(self):
        path = self.root / 'test.jsonl'
        rows = [{'question': f'question {i}', 'answer': f'answer {i}\n#### {i}'} for i in range(33)]
        source = '\n' + '\n'.join(json.dumps(row) for row in rows)
        path.write_text(source)
        text = gsm8k_corpus(path)
        self.assertEqual(text.count('Question: '), 32)
        self.assertIn('answer 31\n#### 31', text)
        self.assertNotIn('question 32', text)
        self.assertEqual(path.read_text(), source)

    def test_gsm8k_json_and_schema_negatives(self):
        path = self.root / 'test.json'
        path.write_text(json.dumps([{'question': 'How many?', 'answer': 'Two.\n#### 2'}]))
        self.assertEqual(gsm8k_corpus(path), 'Question: How many?\nAnswer: Two.\n#### 2\n\n')
        for value in ([], {'test': []}, [{'question': 'Q'}], [{'question': 'Q', 'answer': None}]):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                gsm8k_corpus(path)

    def test_gsm8k_parquet(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('optional pyarrow is not installed')
        path = self.root / 'test.parquet'
        pq.write_table(pa.table({'question': ['How many?'], 'answer': ['Two.\n#### 2']}), path)
        self.assertEqual(gsm8k_corpus(path), 'Question: How many?\nAnswer: Two.\n#### 2\n\n')


if __name__ == '__main__':
    unittest.main()
