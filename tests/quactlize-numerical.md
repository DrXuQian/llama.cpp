# K-pack model numerical comparison

Run `bash tests/run-quactlize-numerical.sh` from the PPU checkout after the
cache smoke passes. Set `MODEL`, `PPU_SDK`, `QUACTLIZE_PPU_BUNDLE`,
`QUACTLIZE_PPU_PACK_LIBRARY`, `CACHE_DIR` and the existing `BUILD_DIR`.
Use `--help` for optional variables. The runner downloads the usual WikiText-2
test corpus with the repository's existing script unless `EVAL_FILE` is set.
Corpus download requires network access, wget/curl and unzip.

For an offline GSM8K dataset, set `GSM8K_FILE=/path/to/test.jsonl` instead of
`EVAL_FILE`. JSONL, JSON arrays and Parquet with `question`/`answer` columns
are supported; Parquet requires an already installed `pyarrow`. No package
is installed by this runner. Prefer the test split; no split is selected
implicitly from a dataset directory. The first 32 records are rendered as
fixed question/answer text and all routes consume that same text. The source
file is left unchanged, and source/text SHA-256 receipts are saved. This
bypasses the download entirely. It measures likelihood differences on the
sample, **not GSM8K generated-answer accuracy**. Scored coverage remains two
256-token chunks, not necessarily all 32 questions.

Only the existing `llama-perplexity` target is built incrementally. The
Quactlize libraries, weights, cache format and compute implementation do not
change. A complete cache is reused; no new background cache write is needed.
Each invocation is first parsed with `--help`, before the profiler/model run.
Perplexity uses the common `--log-colors off` option, not completion's
example-specific `--color` option.

The first integration check uses two 256-token chunks, with 254 scored tokens
per mode. It runs both batch/ubatch 128 (prefill) and 1 (teacher-forced decode).
It is not a full-corpus or long-context accuracy qualification. Other layers
remain on the ordinary GPU backend: the override covers the model's grouped
expert weights, not every operation in the model.

Each mode runs five fresh processes:

| Run | GPU path | Purpose |
| --- | --- | --- |
| reference-save | Ordinary GGUF weights | Save reference log probabilities |
| reference-self | Ordinary GGUF weights | Measure replay/serialization noise |
| kpack-save | GPU pack, no disk cache | Save K-pack log probabilities |
| cache-self | Cached K-pack | Compare persistence against uncached K-pack |
| cache-reference | Cached K-pack | Compare compute route against ordinary GPU |

The saved log probabilities use the existing tool's compressed format.
Self-comparison is not required to have exactly zero KLD. PPL, mean/max KLD,
PPL ratio and top-token agreement require numerical review; finite values
alone do not establish accuracy admission. No CPU model/packer reference is run.

## Kernel execution evidence

All five processes run under the SDK's `asys` kernel-activity tracer, with
model warmup disabled and normal CUDA graph execution retained. The runner
exports the activity to SQLite and matches actual executed device symbols
against `hgobjdump --list-elf` from the supplied Quactlize format libraries.
The matching inventory includes grouped GEMM device entry points only.
Packing, metadata prepass, gather/scatter, host API calls and DSO loading do
not qualify as GEMM execution evidence.

The ordinary GPU baseline must have nonempty GPU activity and zero matching
Quactlize GEMMs. K-pack runs must have matching GEMM activity and the requested
batch's route records for every grouped tensor in the cache manifest, after
evaluation starts. The current ABI launches one grouped GEMM per tensor and
batch: the trace must contain at least that many GEMMs for the 512 input
tokens, not just one warmup kernel. A missing trace or contradictory route fails.
Kernel symbols shared across libraries list all candidate libraries; those
names alone do not resolve which DSO instance owns an identical symbol.

This is instrumentation for correctness/path verification, not a performance
measurement. Tracing can change timing. Do not compare these process times
with the unprofiled cache smoke.

Upload the printed `llama-kpack-numerical.*.results.tgz`. It contains compact
numerical summaries, device-symbol/call-count evidence and raw application
logs. The potentially large `.asysrep`, SQLite exports and saved probability
payloads stay on the box. The reports can be opened in Asight for inspection.

Host parser checks: `python3 tests/test-quactlize-numerical.py`.

## Extended numerical and unprofiled timing pass

After the short gate, run the same entry with `--extended`. Required inputs
are unchanged. This is a fixed follow-up suite, not a new config sweep or
kernel build. With both default batches it runs 16 fresh processes:

| Per batch | Coverage | Profiler | Purpose |
| --- | --- | --- | --- |
| cache-proof | 2 chunks x 256 | Kernel activity | Reconfirm the delivered GEMMs execute |
| reference-save | 8 chunks x 1024 | None | Save ordinary GPU probabilities |
| reference-self | Same saved input | None | Quantify replay/serialization noise |
| cache-reference | Same saved input | None | Compare cached K-pack against ordinary GPU |
| reference, cache, cache, reference | 8 chunks x 1024 each | None | Separate model-timer samples (ABBA) |

Numerical coverage is 8,192 input tokens and 4,088 scored positions per mode,
about 16 times the short gate. GSM8K uses the first 256 records to supply the
text; this still is not generated-answer accuracy or necessarily all 256
questions. Too few input tokens, mismatched headers or incomplete saved
payloads reject instead of silently shrinking the sample. Batch 1 and 128
must save identical input token receipts. The already checked GPU-pack and
cache-persistence comparison is not repeated; every K-pack run uses the
existing cache, requiring zero misses. No new cache is written.

Only the short proof is traced. The extended numerical logs still require
post-start route coverage, but correctly report `kernel_execution=NOT_COLLECTED`
and null kernel counts; a host route record is not a new device trace. Short
proofs and full numerical runs have separate results and share the same
build/library receipts. The default short suite retains its trace requirements.

Performance uses separate perplexity processes with normal model warmup,
`--verbosity 4`, no profiler, no probability save and no KL comparison.
The common logger maps library INFO callbacks to threshold 4, unlike the
application's INFO macros at 3; threshold 3 suppresses model timers and
buffer/cache receipts. Level 4 still excludes DEBUG (5) route/graph logs;
it does not enable an activity profiler.
Reported throughput uses `llama_perf_context_print` model-evaluation timers,
not process wall time. Loading, CPU likelihood calculation and probability
file I/O must not be presented as GEMM time. This is model throughput with
the remaining backend work included, not isolated GEMM latency. Batch 128
uses the 8,192-token prompt timer; batch 1 uses the 4,088 scored single-token
runs (the tool classifies the preceding context work as prompt evaluation).
The exact timer counts and warmup are checked. These low-verbosity processes
check buffer placement/cache receipts; their execution evidence is the
separate short proof, not their own route trace.

`performance-summary.json` retains both samples per arm, median/min/max,
spread and relative throughput change. Two ABBA samples per arm are an
initial comparison, not a confidence bound or a guaranteed 5% precision.
Numerical and performance admissions both remain review-based.

Allow at least 8 GiB free in `RESULT_ROOT`. For the 248,320-token vocabulary,
the two reference probability files occupy about 3.8 GiB total; short traces
need additional space. Large payloads remain on the box; upload only the
printed `.results.tgz`, including `summary.tsv`, `performance-summary.json`
and the original logs. GPU results are still required; host fixture tests
exercise shell phase order/arguments/parsers, not numerical correctness.

If numerical stages have completed but performance needs repeating, use
`--performance-only` with the same inputs. It creates a new result directory
and runs only the ABBA timing processes; no probability files or device traces
are produced, and accuracy is marked `NOT_RUN`. `EVAL_BATCHES=128` limits this
to four prefill processes. Preserve the previous numerical archive separately;
this is a scoped rerun, not an automatic resume or merging of evidence.
