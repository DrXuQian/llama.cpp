#!/usr/bin/env bash
# Model arithmetic and actual PPU kernel execution, using the existing GPU reference.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 1
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' \
        'Required: MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY CACHE_DIR' \
        'Optional: BUILD_DIR=build-ppu RESULT_ROOT=/workspace JOBS=192 CUDA_VISIBLE_DEVICES=0' \
        'Optional: EVAL_FILE=<text corpus>; otherwise download the standard WikiText-2 test corpus.' \
        'Offline alternative: GSM8K_FILE=<local JSONL/JSON/Parquet file>, exclusive with EVAL_FILE.' \
        'Optional: EVAL_BATCHES="128 1" ASYS=<SDK asys executable>' \
        'Requires an existing configured PPU build and a complete cache from the cache smoke.' \
        'Only llama-perplexity is incrementally built. No Quactlize library rebuild.' \
        'Two 256-token chunks per mode; numerical admission requires result review.'
    exit 0
fi
set +u
set -Ee -o pipefail
stage=precheck
RUN=
finish() {
    local rc=$?
    trap - ERR EXIT
    if [[ -n $RUN && -d $RUN/results ]]; then
        printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
        if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
            printf '\nresults=%s.results.tgz\n' "$RUN"
        else
            printf '\nARCHIVE_FAILED logs=%s/results\n' "$RUN" >&2
        fi
        printf 'raw_traces=%s/traces\n' "$RUN"
    fi
    printf 'runner_rc=%s stage=%s\n' "$rc" "$stage"
}
trap 'printf "FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
trap finish EXIT
for key in MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY CACHE_DIR; do
    [[ -n ${!key:-} ]] || { printf 'MISSING %s\n' "$key" >&2; false; }
done
BUILD_DIR=${BUILD_DIR:-build-ppu}
RESULT_ROOT=${RESULT_ROOT:-/workspace}
JOBS=${JOBS:-192}
[[ $JOBS =~ ^[1-9][0-9]*$ ]]
if [[ -n ${GSM8K_FILE:-} ]]; then
    [[ -z ${EVAL_FILE:-} ]] || { printf 'Set only one of GSM8K_FILE and EVAL_FILE.\n' >&2; false; }
    [[ -f $GSM8K_FILE && -s $GSM8K_FILE && -r $GSM8K_FILE ]]
fi
read -ra BATCHES <<< "${EVAL_BATCHES:-128 1}"
[[ ${#BATCHES[@]} -gt 0 && ${#BATCHES[@]} -le 2 ]]
for batch in "${BATCHES[@]}"; do [[ $batch == 1 || $batch == 128 ]]; done
[[ ${#BATCHES[@]} == 1 || ${BATCHES[0]} != "${BATCHES[1]}" ]]
[[ -d $RESULT_ROOT && -s $BUILD_DIR/CMakeCache.txt && -s $CACHE_DIR/manifest.json ]]
REPO=$(pwd -P)
[[ -f $REPO/tests/quactlize_numerical.py && -f $REPO/scripts/get-wikitext-2.sh ]]
[[ -r $MODEL && -r $QUACTLIZE_PPU_PACK_LIBRARY && -r $QUACTLIZE_PPU_BUNDLE/manifest.json ]]
git diff --quiet
git diff --cached --quiet
source "$PPU_SDK/envsetup.sh"
set -Ee -o pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY
export LC_ALL=C
unset LLAMA_ARG_KPACK_CACHE
unset GGML_CUDA_DISABLE_GRAPHS
ASYS=${ASYS:-$PPU_SDK/asight/bin/asys}
[[ -x $ASYS && -x $PPU_SDK/bin/hgobjdump && -x /usr/bin/time ]]
for tool in python3 c++filt cmake; do command -v "$tool" >/dev/null; done
grep -qx 'GGML_USE_PPU:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
grep -qx 'GGML_NCP_QUACTLIZE:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
for option in FA MOE GDN; do
    grep -qx "GGML_NCP_$option:BOOL=OFF" "$BUILD_DIR/CMakeCache.txt"
done

RUN=$(mktemp -d "$RESULT_ROOT/llama-kpack-numerical.XXXXXX")
mkdir "$RUN/results" "$RUN/traces" "$RUN/logprobs"
printf 'RUN=%s\n' "$RUN"
git rev-parse HEAD > "$RUN/results/source.txt"
cp "$CACHE_DIR/manifest.json" "$RUN/results/cache-manifest.json"
cp "$QUACTLIZE_PPU_BUNDLE/manifest.json" "$RUN/results/bundle-manifest.json"
grep -E '^GGML_(USE_PPU|NCP_|CUDA_GRAPH)' "$BUILD_DIR/CMakeCache.txt" > "$RUN/results/build-options.txt"
"$ASYS" --version > "$RUN/results/asys-version.txt" 2>&1
stage=corpus
if [[ -n ${GSM8K_FILE:-} ]]; then
    mkdir "$RUN/corpus"
    sha256sum "$GSM8K_FILE" > "$RUN/results/corpus-source.sha256"
    python3 tests/quactlize_numerical.py gsm8k "$GSM8K_FILE" > "$RUN/corpus/gsm8k.txt"
    EVAL_FILE="$RUN/corpus/gsm8k.txt"
    printf 'KPACK_CORPUS source=local-gsm8k sample=first-32 purpose=likelihood-comparison answer_accuracy=NOT_MEASURED\n' \
        | tee "$RUN/results/corpus-source.log"
elif [[ -z ${EVAL_FILE:-} ]]; then
    mkdir "$RUN/corpus"
    if (cd "$RUN/corpus" && timeout 180s bash "$REPO/scripts/get-wikitext-2.sh") > "$RUN/results/corpus-download.log" 2>&1; then
        printf 'KPACK_CORPUS source=wikitext-2\n' | tee "$RUN/results/corpus-source.log"
    else
        download_rc=$?
        tail -n 25 "$RUN/results/corpus-download.log" >&2
        printf 'Corpus download failed; set EVAL_FILE or GSM8K_FILE to use existing local data.\n' >&2
        exit "$download_rc"
    fi
    EVAL_FILE="$RUN/corpus/wikitext-2-raw/wiki.test.raw"
else
    printf 'KPACK_CORPUS source=local-text\n' | tee "$RUN/results/corpus-source.log"
fi
[[ -s $EVAL_FILE && -r $EVAL_FILE ]]
sha256sum "$EVAL_FILE" > "$RUN/results/corpus.sha256"
stage=inventory
python3 tests/quactlize_numerical.py inventory "$QUACTLIZE_PPU_BUNDLE" "$PPU_SDK/bin/hgobjdump" \
    > "$RUN/results/kernel-inventory.json"
sha256sum "$QUACTLIZE_PPU_PACK_LIBRARY" > "$RUN/results/pack-library.sha256"
stage=build
cmake --build "$BUILD_DIR" --target llama-perplexity -j "$JOBS" 2>&1 | tee "$RUN/results/build.log"
sha256sum "$BUILD_DIR/bin/llama-perplexity" "$BUILD_DIR/bin/libggml-cuda.so" > "$RUN/results/binaries.sha256"

for batch in "${BATCHES[@]}"; do
    # No model warmup: trace evidence must come from the scored workload,
    # not a dummy warmup invocation. CUDA graph execution remains enabled.
    ARGS=(-m "$MODEL" --mmap -ngl 99 --split-mode none --fit off --no-warmup
        -c 256 -b "$batch" -ub "$batch" -t 16 -tb 32 --chunks 2 --log-colors off -f "$EVAL_FILE" -v)
    for phase in reference-save reference-self kpack-save cache-self cache-reference; do
        stage="b$batch-$phase"
        case "$phase" in
            reference-save) EXTRA=(-ot 'ffn_.*_exps=CUDA0'
                --save-all-logits "$RUN/logprobs/b$batch-reference");;
            reference-self) EXTRA=(-ot 'ffn_.*_exps=CUDA0' --kl-divergence
                --kl-divergence-base "$RUN/logprobs/b$batch-reference");;
            kpack-save) EXTRA=(-ot 'ffn_.*_exps=CUDA0_KPACK'
                --save-all-logits "$RUN/logprobs/b$batch-kpack");;
            cache-self) EXTRA=(-ot 'ffn_.*_exps=CUDA0_KPACK' --kpack-cache "$CACHE_DIR"
                --kl-divergence --kl-divergence-base "$RUN/logprobs/b$batch-kpack");;
            cache-reference) EXTRA=(-ot 'ffn_.*_exps=CUDA0_KPACK' --kpack-cache "$CACHE_DIR"
                --kl-divergence --kl-divergence-base "$RUN/logprobs/b$batch-reference");;
        esac
        # Parse the exact invocation before starting the profiler or loading weights.
        if ! "$BUILD_DIR/bin/llama-perplexity" "${ARGS[@]}" "${EXTRA[@]}" --help \
                > "$RUN/results/$stage.cli.log" 2>&1; then
            tail -n 25 "$RUN/results/$stage.cli.log" >&2
            printf 'KPACK_NUMERICAL FAIL phase=%s reason=CLI_ARGUMENTS\n' "$stage" >&2
            false
        fi
        printf '\nKPACK_NUMERICAL_RUN phase=%s profiler=kernel-activity\n' "$stage"
        /usr/bin/time -v -o "$RUN/results/$stage.time" \
            "$ASYS" profile --trace hggc --hggc-trace-set kernel-activity --sample none \
            --kill none --show-output true --output "$RUN/traces/$stage.asysrep" \
            "$BUILD_DIR/bin/llama-perplexity" "${ARGS[@]}" "${EXTRA[@]}" </dev/null \
            2>&1 | tee "$RUN/results/$stage.log"
        "$ASYS" export --output "$RUN/traces/$stage.sqlite" "$RUN/traces/$stage.asysrep" \
            > "$RUN/results/$stage.export.log" 2>&1
        python3 tests/quactlize_numerical.py check "$RUN/results/$stage.log" "$RUN/traces/$stage.sqlite" \
            "$RUN/results/kernel-inventory.json" "$RUN/results/cache-manifest.json" \
            --batch "$batch" --phase "$phase" > "$RUN/results/$stage.json" \
            2> "$RUN/results/$stage.check.log" || { cat "$RUN/results/$stage.check.log" >&2; false; }
        python3 - "$RUN/results/$stage.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('KPACK_KERNEL_EXECUTION batch={} phase={} verdict={} grouped_gemm_calls={} routed_tensors={}'.format(
    r['batch'], r['phase'], r['route_verdict'], r['quactlize_grouped_gemm_calls'], r['routed_tensors']))
print('KPACK_NUMERICAL_METRICS ' + json.dumps(r['metrics'], sort_keys=True))
PY
    done
    for kind in reference kpack; do
        python3 tests/quactlize_numerical.py logprobs "$RUN/logprobs/b$batch-$kind" \
            > "$RUN/results/b$batch-$kind-tokens.json"
    done
    cmp "$RUN/results/b$batch-reference-tokens.json" "$RUN/results/b$batch-kpack-tokens.json"
done
stage=done
python3 - "$RUN/results" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
metrics = ('ppl', 'ppl_ratio', 'mean_kld', 'max_kld', 'same_top_pct')
with (root / 'summary.tsv').open('w') as out:
    writer = csv.writer(out, delimiter='\t', lineterminator='\n')
    writer.writerow(('batch', 'phase', 'route', 'grouped_gemm_calls', *metrics))
    for file in sorted(root.glob('b*-*.json')):
        row = json.loads(file.read_text())
        if 'phase' not in row:
            continue
        writer.writerow((row['batch'], row['phase'], row['route_verdict'],
                         row['quactlize_grouped_gemm_calls'], *(row['metrics'].get(m, '') for m in metrics)))
PY
printf 'KPACK_MODEL_NUMERICAL COMPLETE kernel_execution=PASS accuracy=PENDING_REVIEW perf=NOT_MEASURED\n' \
    | tee "$RUN/results/verdict.log"
