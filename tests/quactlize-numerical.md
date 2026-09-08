# K-pack model numerical comparison

Run `bash tests/run-quactlize-numerical.sh` from the PPU checkout after the
cache smoke passes. Set `MODEL`, `PPU_SDK`, `QUACTLIZE_PPU_BUNDLE`,
`QUACTLIZE_PPU_PACK_LIBRARY`, `CACHE_DIR` and the existing `BUILD_DIR`.
Use `--help` for optional variables. The runner downloads the usual WikiText-2
test corpus with the repository's existing script unless `EVAL_FILE` is set.
Corpus download requires network access, wget/curl and unzip.

Only the existing `llama-perplexity` target is built incrementally. The
Quactlize libraries, weights, cache format and compute implementation do not
change. A complete cache is reused; no new background cache write is needed.

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
