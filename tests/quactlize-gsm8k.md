# Single-request GSM8K answer comparison

Run `bash tests/run-quactlize-gsm8k.sh` with the same model, PPU SDK, runtime
bundle, pack library, existing cache and configured build as the numerical
gate. Also set `GSM8K_FILE` to the local **test** JSONL/JSON/Parquet file.
Parquet uses the existing `pyarrow` dependency; JSON/JSONL needs only the
Python standard library. Nothing is downloaded.

This is generated-answer accuracy, not perplexity or token agreement.
The default is a deterministic 128-question sample without replacement
(seed 20260908), not a full GSM8K benchmark or a claim of equivalence with
published benchmark protocols. Each question is sent without its gold answer
or worked solution. Both arms use the model's own chat template, thinking off,
greedy decoding, and the same instruction to reason and end with `#### <number>`.
Correctness compares the normalized final number exactly, with no model grader.
Numbers elsewhere in the reasoning are not accepted as final answers.

The ordinary GPU server runs first, then a fresh cached K-pack server. Each
model loads **once per arm** and answers sequentially with a single server
slot and one outstanding request. Business request batch size is **1**.
`GSM8K_TOKEN_BATCH=128` controls the single request's prefill token chunk;
it is not 128 concurrent sequences. Generation proceeds token by token.
Prompt KV reuse is disabled between questions; the persistent weight cache
is separate and must be a full disk-cache hit without repacking.

The runner formats each prompt through `/apply-template`, tokenizes it,
then calls `/completion` with the exact token array. It checks identical
tokens/settings across arms, one slot, context, effective sampling settings,
prompt counts and no prompt-cache reuse. Only a localhost server started
by the runner is used; an ephemeral alias/key prevents attaching to a
different process. It terminates only its own server child at the end.

Defaults: context 4096, `GSM8K_MAX_TOKENS=1024`, `GSM8K_CASES=128`.
The prompt plus cap must fit before execution. Length-limited or context-
truncated answers count as incorrect even if they contain the gold number.
Unparseable answers also stay in the denominator and are counted separately.
HTTP/runtime failures stop the run as incomplete and preserve all completed
raw responses; they do not produce an accuracy pass or a smaller denominator.
This initial subset needs review, especially if truncation/parser failures
are frequent. Use a larger cap/sample for a follow-up, not a retroactive
change to the scoring of a completed run.

Only `llama-server` and its changed dependencies are built incrementally.
No Quactlize DSO, kernel, loader, ABI or config selector changes are made.
Existing builds must enable PPU/Quactlize and disable unrelated FA/MOE/GDN
external libraries. The Web UI build is disabled to avoid asset downloads.

There is **no profiler** in this suite. Logs verify ordinary GPU versus
K-pack weight placement and full cache hits; this is not a new device-kernel
trace. The earlier numerical gate supplies that independent execution
evidence. The summary explicitly says `kernel_execution=NOT_COLLECTED`.
Per-request server/wall times are retained for diagnosis but an answer A/B,
whose output lengths may differ, is not a controlled throughput benchmark.

Progress is printed for every answer, with an observed estimate for the
remaining questions **in the current arm**, excluding the other arm/load.
There are 256 total requests at the default sample size. No full-duration
guarantee is inferred from isolated GEMM timings.

Upload the printed `llama-kpack-gsm8k.*.results.tgz`. It includes:

- `summary.json`: both accuracies, percentage-point difference, truncation/
  parse counts and both directions of paired correctness changes;
- `protocol.json`: exact dataset hash, selected row IDs, question-only prompts
  and gold answers used only by the scorer;
- `reference.jsonl`, `kpack.jsonl`: actual input tokens, full generated text,
  raw server responses, effective settings and timing;
- server logs, placement/cache receipts, build/library hashes and runner status.

Host checks (no model or device needed):

```sh
python3 tests/test-quactlize-gsm8k.py
python3 tests/test-quactlize-numerical.py
bash -n tests/run-quactlize-gsm8k.sh
```
