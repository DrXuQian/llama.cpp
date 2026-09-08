#!/usr/bin/env bash
# One model/server per arm; one outstanding generated-answer request at a time.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 1
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' \
        'Required: MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY CACHE_DIR GSM8K_FILE' \
        'Optional: BUILD_DIR=build-ppu RESULT_ROOT=/workspace JOBS=192 CUDA_VISIBLE_DEVICES=0' \
        'Optional: GSM8K_CASES=128 GSM8K_MAX_TOKENS=1024 GSM8K_TOKEN_BATCH=128' \
        'Business request batch is always 1. Token batch controls only prompt processing.' \
        'Fixed seed, greedy, thinking off, 4096-token context, question-only zero-shot prompts.' \
        'Local JSONL/JSON/Parquet only; no downloads, CPU reference or profiler.' \
        'Requires a complete existing K-pack cache and a configured PPU build.' \
        'Incrementally builds llama-server, without rebuilding Quactlize DSOs.'
    exit 0
fi
set +u
set -Ee -o pipefail
[[ $# == 0 ]]
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
    fi
    printf 'runner_rc=%s stage=%s\n' "$rc" "$stage"
}
trap 'printf "FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
trap finish EXIT
for key in MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY CACHE_DIR GSM8K_FILE; do
    [[ -n ${!key:-} ]] || { printf 'MISSING %s\n' "$key" >&2; false; }
done
BUILD_DIR=${BUILD_DIR:-build-ppu}
RESULT_ROOT=${RESULT_ROOT:-/workspace}
JOBS=${JOBS:-192}
GSM8K_CASES=${GSM8K_CASES:-128}
GSM8K_MAX_TOKENS=${GSM8K_MAX_TOKENS:-1024}
GSM8K_TOKEN_BATCH=${GSM8K_TOKEN_BATCH:-128}
for number in "$JOBS" "$GSM8K_CASES" "$GSM8K_MAX_TOKENS"; do [[ $number =~ ^[1-9][0-9]*$ ]]; done
[[ $GSM8K_TOKEN_BATCH == 1 || $GSM8K_TOKEN_BATCH == 128 ]]
[[ $GSM8K_MAX_TOKENS -lt 4096 ]]
[[ -d $RESULT_ROOT && -s $BUILD_DIR/CMakeCache.txt && -s $CACHE_DIR/manifest.json ]]
[[ -r $MODEL && -s $GSM8K_FILE && -r $QUACTLIZE_PPU_PACK_LIBRARY && -r $QUACTLIZE_PPU_BUNDLE/manifest.json ]]
[[ -f tests/quactlize_gsm8k.py ]]
git diff --quiet
git diff --cached --quiet
source "$PPU_SDK/envsetup.sh"
set -Ee -o pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY LC_ALL=C
for tool in python3 cmake; do command -v "$tool" >/dev/null; done
grep -qx 'GGML_USE_PPU:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
grep -qx 'GGML_NCP_QUACTLIZE:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
for option in FA MOE GDN; do grep -qx "GGML_NCP_$option:BOOL=OFF" "$BUILD_DIR/CMakeCache.txt"; done
python3 - "$GSM8K_FILE" "$GSM8K_CASES" <<'PY'
import pathlib, sys
sys.path.insert(0, 'tests')
from quactlize_gsm8k import prepare
cases = prepare(pathlib.Path(sys.argv[1]), int(sys.argv[2]), 20260908)
print(f'KPACK_GSM8K_DATASET questions={len(cases)} gold=VALID prompts=QUESTION_ONLY')
PY
RUN=$(mktemp -d "$RESULT_ROOT/llama-kpack-gsm8k.XXXXXX")
mkdir "$RUN/results"
printf 'RUN=%s\n' "$RUN"
git rev-parse HEAD > "$RUN/results/source.txt"
cp "$CACHE_DIR/manifest.json" "$RUN/results/cache-manifest.json"
cp "$QUACTLIZE_PPU_BUNDLE/manifest.json" "$RUN/results/bundle-manifest.json"
sha256sum "$QUACTLIZE_PPU_PACK_LIBRARY" > "$RUN/results/libraries.sha256"
for fmt in 0 1 2 3 4; do
    sha256sum "$QUACTLIZE_PPU_BUNDLE/libquactlize_ppu_fmt$fmt.so" >> "$RUN/results/libraries.sha256"
done
stage=build-server
printf 'KPACK_GSM8K_BUILD target=llama-server jobs=%s quactlize_dso_rebuild=0\n' "$JOBS"
cmake -S . -B "$BUILD_DIR" -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    2>&1 | tee "$RUN/results/configure.log"
cmake --build "$BUILD_DIR" --target llama-server -j "$JOBS" 2>&1 | tee "$RUN/results/build.log"
grep -E '^GGML_(USE_PPU|NCP_|CUDA_GRAPH)' "$BUILD_DIR/CMakeCache.txt" > "$RUN/results/build-options.txt"
sha256sum "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/libggml-cuda.so" \
    "$BUILD_DIR/bin/libllama.so" > "$RUN/results/binaries.sha256"
stage=answer-evaluation
python3 -u tests/quactlize_gsm8k.py --dataset "$GSM8K_FILE" --output "$RUN/results" \
    --binary "$BUILD_DIR/bin/llama-server" --model "$MODEL" --cache "$CACHE_DIR" \
    --cases "$GSM8K_CASES" --max-tokens "$GSM8K_MAX_TOKENS" --batch "$GSM8K_TOKEN_BATCH" \
    2>&1 | tee "$RUN/results/evaluation.log"
stage=done
printf 'KPACK_GSM8K COMPLETE accuracy=PENDING_REVIEW kernel_trace=NOT_COLLECTED\n'
