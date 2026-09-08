#!/usr/bin/env python3
"""Host checks for the numerical runner's evidence parser, not a model oracle."""

import copy
import json
from pathlib import Path
import re
import shlex
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

from quactlize_numerical import activity, analyze, analyze_log, gsm8k_corpus, logprobs, performance, performance_summary


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

    def check(self, phase="cache-reference", text=None, manifest=None, batch=1):
        self.log.write_text(self.text if text is None else text)
        return analyze(self.log, self.db, self.index, manifest or self.manifest, batch, phase)

    def save_text(self, batch):
        source = (Path(__file__).parents[1] / 'tools/perplexity/perplexity.cpp').read_text()
        fmt = re.search(r'LOG_INF\("([^"\n]*calculating perplexity[^"\n]*)", __func__', source)
        self.assertIsNotNone(fmt)
        header = fmt[1].replace('\\n', '\n') % ('perplexity', 2, 256, batch, 1)
        route = f'[ncp-route] MUL_MAT_ID blk.0.ffn_up_exps.weight T={batch} E=256 used=8 -> so-quactlize-kpack\n'
        return header + route + 'Final estimate: PPL = 7.1250 +/- 0.01000\n'

    def test_kpack_save_producer_message(self):
        for batch in (1, 128):
            with self.subTest(batch=batch):
                result = self.check('kpack-save', self.save_text(batch), batch=batch)
                self.assertEqual(result['metrics']['ppl'], 7.125)

    def test_reference_save_producer_message(self):
        self.edit_db("UPDATE StringIds SET value='ordinary_gpu_gemm' WHERE id IN (1, 2)")
        for batch in (1, 128):
            with self.subTest(batch=batch):
                result = self.check('reference-save', self.save_text(batch).replace('so-quactlize-kpack', 'GENERIC'), batch=batch)
                self.assertEqual(result['quactlize_grouped_gemm_calls'], 0)

    def test_save_still_rejects_wrong_coverage(self):
        for before, after in (('over 2 chunks', 'over 1 chunks'), ('n_ctx=256', 'n_ctx=512'),
                              ('batch_size=128', 'batch_size=1'), ('n_seq=1', 'n_seq=2'), ('n_seq=1', 'n_seq=10')):
            with self.subTest(field=before), self.assertRaisesRegex(ValueError, 'context/chunks/batch'):
                self.check('kpack-save', self.save_text(128).replace(before, after), batch=128)

    def test_save_requires_save_header(self):
        with self.assertRaisesRegex(ValueError, 'observed='):
            self.check('kpack-save', self.text + 'Final estimate: PPL = 7.1250 +/- 0.01000\n')

    def test_untraced_extended_is_not_device_evidence(self):
        text = self.text.replace('over 2 chunks, n_ctx=256', 'over 8 chunks, n_ctx=1024')
        self.log.write_text(text)
        result = analyze_log(self.log, self.manifest, 1, 'cache-reference', 1024, 8)
        self.assertEqual(result['kernel_execution'], 'NOT_COLLECTED')
        self.assertIsNone(result['quactlize_grouped_gemm_calls'])
        self.assertIsNone(result['kernels'])
        self.assertEqual(result['route_evidence'], 'POST_START_ROUTE_LOG')
        for context, chunks in ((256, 2), (1024, 2), (256, 8), (1, 8)):
            with self.subTest(context=context, chunks=chunks), self.assertRaises(ValueError):
                analyze_log(self.log, self.manifest, 1, 'cache-reference', context, chunks)

    def test_extended_trace_needs_extended_call_count(self):
        self.log.write_text(self.text.replace('over 2 chunks, n_ctx=256', 'over 8 chunks, n_ctx=1024'))
        with self.assertRaisesRegex(ValueError, 'incomplete evaluation kernel trace'):
            analyze(self.log, self.db, self.index, self.manifest, 1, 'cache-reference', 1024, 8)

    def test_short_cached_proof_still_requires_trace_and_hit(self):
        text = 'cache_uploads=1 resident_misses=0\n' + self.save_text(128)
        result = self.check('cache-proof', text, batch=128)
        self.assertEqual(result['kernel_execution'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'full cache hit'):
            self.check('cache-proof', self.save_text(128), batch=128)
        self.edit_db('DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL')
        with self.assertRaisesRegex(ValueError, 'empty GPU'):
            self.check('cache-proof', text, batch=128)

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

    def test_extended_logprobs_coverage(self):
        path = self.root / 'extended-logprobs'
        payload = b'_logits_' + struct.pack('<iii', 1024, 8, 8) + bytes(8192 * 4) + bytes(4088 * 12 * 2)
        path.write_bytes(payload)
        self.assertEqual(logprobs(path, 1024, 8)['scored_tokens'], 4088)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            logprobs(path)
        path.write_bytes(payload[:-2])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            logprobs(path, 1024, 8)

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

    def test_extended_gsm8k_sample(self):
        path = self.root / 'test.jsonl'
        path.write_text('\n'.join(json.dumps({'question': f'Q{i}', 'answer': f'A{i}'}) for i in range(300)))
        self.assertEqual(gsm8k_corpus(path, 256).count('Question:'), 256)
        self.assertEqual(gsm8k_corpus(path).count('Question:'), 32)
        with self.assertRaisesRegex(ValueError, 'positive'):
            gsm8k_corpus(path, 0)

    def perf_text(self, batch, cached=True):
        header = (f'perplexity: calculating perplexity over 8 chunks, n_ctx=1024, batch_size={batch}, n_seq=1\n'
                  'Final estimate: PPL = 7.1250 +/- 0.01\n')
        timers = ('llama_perf_context_print: prompt eval time = 1000.00 ms / 8192 tokens\n'
                  'llama_perf_context_print:        eval time = 0.00 ms / 1 runs\n') if batch == 128 else (
                  'llama_perf_context_print: prompt eval time = 5000.00 ms / 4104 tokens\n'
                  'llama_perf_context_print:        eval time = 4000.00 ms / 4088 runs\n')
        return ('common_init_from_params: warming up the model with an empty run\n'
                + ('CUDA0_KPACK model buffer size = 16.00 MiB\ncache_uploads=1 resident_misses=0\n' if cached else '')
                + header + timers)

    def test_performance_model_timers_and_no_trace_claim(self):
        for batch in (1, 128):
            for phase in ('cache-perf', 'reference-perf'):
                with self.subTest(batch=batch, phase=phase):
                    self.log.write_text(self.perf_text(batch, cached=phase.startswith('cache')))
                    result = performance(self.log, self.manifest, batch, phase)
                    self.assertEqual(result['performance']['tokens_per_second'], 1022 if batch == 1 else 8192)
                    self.assertEqual(result['kernel_execution'], 'NOT_COLLECTED')
                    self.assertEqual(result['route_evidence'], 'BUFFER_PLACEMENT')
                    self.assertIsNone(result['routed_tensors'])

    def test_performance_rejects_missing_work_and_instrumentation(self):
        replacements = (('warming up the model with an empty run', 'no warmup'),
                        ('8192 tokens', '8191 tokens'), ('1000.00 ms', 'nan ms'),
                        ('1000.00 ms', '0.00 ms'), ('1000.00 ms', '-1000.00 ms'),
                        ('resident_misses=0', 'resident_misses=1'), ('16.00 MiB', '0.00 MiB'))
        text = self.perf_text(128)
        negatives = [text.replace(a, b) for a, b in replacements]
        negatives += [text + extra for extra in ('\n[ncp-route] debug', '\n0.01 D graph reused',
                                                '\nsaving all logits', '\nCUDA error: failed',
                                                '\nprompt eval time = 1000.00 ms / 8192 tokens')]
        for text in negatives:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.log.write_text(text)
                performance(self.log, self.manifest, 128, 'cache-perf')
        self.log.write_text(self.perf_text(1).replace('4088 runs', '4087 runs'))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            performance(self.log, self.manifest, 1, 'cache-perf')

    def test_performance_summary_counts_spread_and_coverage(self):
        records = []
        for phase in ('reference-perf', 'cache-perf', 'cache-perf', 'reference-perf'):
            self.log.write_text(self.perf_text(128, cached=phase.startswith('cache')))
            records.append(performance(self.log, self.manifest, 128, phase))
        records[1]['performance']['tokens_per_second'] *= 1.1
        records[2]['performance']['tokens_per_second'] *= 1.1
        result = performance_summary(records)
        self.assertAlmostEqual(result['rows'][0]['kpack_delta_pct'], 10)
        self.assertEqual(result['admission'], 'PENDING_REVIEW')
        with self.assertRaisesRegex(ValueError, 'two samples'):
            performance_summary(records[:-1])
        mixed = copy.deepcopy(records)
        mixed[0]['chunks'] = 2
        with self.assertRaisesRegex(ValueError, 'different coverage'):
            performance_summary(mixed)
        mixed = copy.deepcopy(records)
        mixed[0]['performance']['profiler'] = True
        with self.assertRaisesRegex(ValueError, 'evidence'):
            performance_summary(mixed)

    def test_runner_phase_orchestration_with_host_fixtures(self):
        # Synthetic CLI/trace payloads test orchestration only, never device admission.
        repo = Path(__file__).parents[1]
        runner = (repo/'tests/run-quactlize-numerical.sh').read_text()
        tail = runner[runner.index('run_phase() {'):]
        timer = self.root/'time'
        timer.write_text(f'#!{sys.executable}\n' + r'''
import pathlib, shlex, subprocess, sys
assert sys.argv[1:3] == ['-v', '-o']
cmd = sys.argv[4:]
rc = subprocess.call(cmd)
pathlib.Path(sys.argv[3]).write_text('Command being timed: '+shlex.join(cmd)+'\nExit status: '+str(rc)+'\n')
sys.exit(rc)
''')
        timer.chmod(0o755)
        self.assertEqual(tail.count('/usr/bin/time'), 1)
        tail = tail.replace('/usr/bin/time', shlex.quote(str(timer)))
        build = self.root/'build'
        (build/'bin').mkdir(parents=True)
        exe = build/'bin/llama-perplexity'
        exe.write_text(f'#!{sys.executable}\n' + r'''
import pathlib, struct, sys
args = sys.argv[1:]
if '--help' in args:
    sys.exit(0)
def arg(name):
    return args[args.index(name)+1]
context, chunks, batch = int(arg('-c')), int(arg('--chunks')), int(arg('-b'))
cached = '--kpack-cache' in args
kpack = '_KPACK' in arg('-ot')
if '--no-warmup' not in args:
    print('warming up the model with an empty run')
if kpack:
    print('CUDA0_KPACK model buffer size = 16.00 MiB')
if cached:
    print('cache_uploads=1 resident_misses=0')
label = 'kl_divergence: computing' if '--kl-divergence' in args else 'perplexity: calculating perplexity'
print(f'{label} over {chunks} chunks, n_ctx={context}, batch_size={batch}, n_seq=1')
if '-v' in args and kpack:
    print(f'[ncp-route] MUL_MAT_ID blk.0.ffn_up_exps.weight T={batch} E=256 used=8 -> so-quactlize-kpack')
if '--kl-divergence' in args:
    base = pathlib.Path(arg('--kl-divergence-base'))
    assert base.is_file()
    print('Mean KLD: 0.000001')
    print('Maximum KLD: 0.000010')
    print('Mean PPL(Q)/PPL(base) : 1.000000')
    print('Same top p: 100.000')
else:
    print('Final estimate: PPL = 7.1250 +/- 0.01')
scored = chunks*(context-1-context//2)
if '--save-all-logits' in args:
    path = pathlib.Path(arg('--save-all-logits'))
    path.write_bytes(b'_logits_'+struct.pack('<iii', context, 8, chunks)+bytes(context*chunks*4)+bytes(scored*24))
if batch == 128:
    print(f'llama_perf_context_print: prompt eval time = 1000.00 ms / {context*chunks} tokens')
    print('llama_perf_context_print: eval time = 0.00 ms / 1 runs')
else:
    print(f'llama_perf_context_print: prompt eval time = 5000.00 ms / {context*chunks-scored} tokens')
    print(f'llama_perf_context_print: eval time = 4000.00 ms / {scored} runs')
''')
        exe.chmod(0o755)
        asys = self.root/'asys'
        asys.write_text(f'#!{sys.executable}\n' + r'''
import json, pathlib, sqlite3, subprocess, sys
args = sys.argv[1:]
out = pathlib.Path(args[args.index('--output')+1])
if args[0] == 'profile':
    pos = next(i for i, a in enumerate(args) if a.endswith('/bin/llama-perplexity'))
    cmd = args[pos:]
    rc = subprocess.call(cmd)
    out.write_text(json.dumps(cmd))
    sys.exit(rc)
cmd = json.loads(pathlib.Path(args[-1]).read_text())
def arg(name):
    return cmd[cmd.index(name)+1]
calls = int(arg('-c'))*int(arg('--chunks'))//int(arg('-b'))
mangled, demangled = ('_kernel', 'exact grouped GEMM') if '_KPACK' in arg('-ot') else ('ordinary', 'ordinary')
with sqlite3.connect(out) as db:
    db.execute('CREATE TABLE StringIds(id INTEGER, value TEXT)')
    db.executemany('INSERT INTO StringIds VALUES(?, ?)', [(1, mangled), (2, demangled)])
    db.execute('CREATE TABLE HGPTI_ACTIVITY_KIND_KERNEL(start INT, end INT, mangledName INT, demangledName INT)')
    db.executemany('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES(?, ?, 1, 2)', [(i*20, i*20+10) for i in range(calls)])
''')
        asys.chmod(0o755)
        for mode in ('smoke', 'extended'):
            with self.subTest(mode=mode):
                run = self.root/mode
                for name in ('results', 'traces', 'logprobs'):
                    (run/name).mkdir(parents=True)
                (run/'results/kernel-inventory.json').write_text(json.dumps(self.index))
                (run/'results/cache-manifest.json').write_text(json.dumps(self.manifest))
                env = {'RUN': str(run), 'MODE': mode, 'BUILD_DIR': str(build), 'MODEL': 'fixture.gguf',
                       'CACHE_DIR': 'fixture-cache', 'EVAL_FILE': 'fixture.txt', 'ASYS': str(asys)}
                script = 'set -Ee -o pipefail\nBATCHES=(128 1)\n'
                script += ''.join(f'{k}={shlex.quote(v)}\n' for k, v in env.items())
                result = subprocess.run(['bash', '-c', script+tail], cwd=repo, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:]+result.stderr)
                phase_files = [p for p in (run/'results').glob('b*.json') if '-tokens' not in p.name]
                self.assertEqual(len(phase_files), 10 if mode == 'smoke' else 16)
                self.assertEqual(len(list((run/'traces').glob('*.sqlite'))), 10 if mode == 'smoke' else 2)
                if mode == 'extended':
                    summary = json.loads((run/'results/performance-summary.json').read_text())
                    self.assertEqual(len(summary['rows']), 2)
                    for p in phase_files:
                        row = json.loads(p.read_text())
                        self.assertEqual(row['kernel_execution'], 'PASS' if row['phase'] == 'cache-proof' else 'NOT_COLLECTED')
                        command = p.with_suffix('.time').read_text().splitlines()[0]
                        self.assertEqual('profile --trace' in command, row['phase'] == 'cache-proof')
                        if row['phase'].endswith('perf'):
                            self.assertNotIn('--no-warmup', command)
                            self.assertNotIn('--save-all-logits', command)
                            self.assertIn('--verbosity 3', command)


if __name__ == '__main__':
    unittest.main()
