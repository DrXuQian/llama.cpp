#!/usr/bin/env bash
# Readiness/model/cache smoke; this does not replace the PPU numeric or performance gates.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run this script with bash; do not source it.\n' >&2
    return 1
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' \
        'Required environment: MODEL, PPU_SDK, QUACTLIZE_PPU_BUNDLE, QUACTLIZE_PPU_PACK_LIBRARY' \
        'Optional: BUILD_DIR=build-ppu RESULT_ROOT=/workspace JOBS=192 CUDA_VISIBLE_DEVICES=0' \
        'Optional numerical comparison: EVAL_FILE=<text corpus, at least 512 tokens>' \
        'EVAL_BATCH=128 for prefill or 1 for teacher-forced decode; results require review.' \
        'Run with bash from the llama.cpp checkout. Existing build artifacts are reused.'
    exit 0
fi

set +u
set -Ee -o pipefail
stage=precheck
RUN=
fail() {
    local rc=$?
    printf '\nFAIL stage=%s line=%s rc=%s\n' "$stage" "${BASH_LINENO[0]}" "$rc" >&2
}
finish() {
    local rc=$?
    trap - ERR EXIT
    set +e
    if [[ -n $RUN && -d $RUN/results ]]; then
        if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
            printf '\nresults=%s\ncache=%s\n' "$RUN.results.tgz" "$RUN/cache"
        else
            printf 'ARCHIVE_FAILED logs=%s/results\n' "$RUN"
        fi
    fi
    printf 'runner_rc=%s stage=%s\n' "$rc" "$stage"
}
trap fail ERR
trap finish EXIT

for key in MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY; do
    if [[ -z ${!key:-} ]]; then
        printf 'MISSING environment: %s\n' "$key" >&2
        false
    fi
done
SDK=$PPU_SDK
BUILD_DIR=${BUILD_DIR:-build-ppu}
RESULT_ROOT=${RESULT_ROOT:-/workspace}
JOBS=${JOBS:-192}
[[ $JOBS =~ ^[1-9][0-9]*$ ]]
EVAL_BATCH=${EVAL_BATCH:-128}
[[ $EVAL_BATCH == 128 || $EVAL_BATCH == 1 ]]
if [[ -n ${EVAL_FILE:-} ]]; then [[ -r $EVAL_FILE && -s $EVAL_FILE ]]; fi
[[ -d $RESULT_ROOT ]]
[[ -f ggml/src/ggml-cuda/quactlize-buft.cu ]]
git diff --quiet
git diff --cached --quiet
for file in "$MODEL" "$SDK/envsetup.sh" "$SDK/CUDA_SDK/bin/nvcc" \
            "$QUACTLIZE_PPU_BUNDLE/manifest.json" "$QUACTLIZE_PPU_PACK_LIBRARY" /usr/bin/time; do
    if [[ ! -r $file ]]; then
        printf 'MISSING file: %s\n' "$file" >&2
        false
    fi
done
[[ -x /usr/bin/time && -x $SDK/CUDA_SDK/bin/nvcc ]]
source "$SDK/envsetup.sh"
set -Ee -o pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY
unset LLAMA_ARG_KPACK_CACHE

RUN=$(mktemp -d "$RESULT_ROOT/llama-kpack-smoke.XXXXXX")
[[ -d $RUN && -n $RUN ]]
mkdir "$RUN/results"
printf 'RUN=%s\n' "$RUN"
git rev-parse HEAD | tee "$RUN/results/source.txt"

stage=libraries
(
    cd "$QUACTLIZE_PPU_BUNDLE"
    printf '%s\n' \
        '46fc3096e1a14b712ad5d7a50de096d2a973ad5826aa3ffe6a6764d1fc12180d  manifest.json' \
        '3aa5487f0b3d325db2c1dc3cb863d733157dc332236cea0f8c178d3f97c9ac24  libquactlize_ppu_fmt0.so' \
        '70d009cdbd93ef6237647e1e22910af3ebe3943664734343cbcb00ec03f712a7  libquactlize_ppu_fmt1.so' \
        '161ce1199c771940afde9d40cb8a2ed092a215a21a1a1afd4312bee9fd3decfa  libquactlize_ppu_fmt2.so' \
        '295c3cef9c07d849afda44c552b2830db616023784942d8f42de229ec7aa830b  libquactlize_ppu_fmt3.so' \
        '807a92cd5b9508a69bfc70b0be3b30f29f50329ecd0644b260c70cc5d27c8ba2  libquactlize_ppu_fmt4.so' |
        sha256sum -c -
) | tee "$RUN/results/libraries.log"
printf '%s  %s\n' \
    611ec98c4315748e4504082aa3071ae9713ee78958cd09b6baaee770b28a3184 \
    "$QUACTLIZE_PPU_PACK_LIBRARY" | sha256sum -c - | tee -a "$RUN/results/libraries.log"

stage=configure
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$SDK/CUDA_SDK/bin/nvcc" \
    -DCUDAToolkit_ROOT="$SDK/CUDA_SDK" -DCMAKE_CUDA_ARCHITECTURES=OFF \
    -DGGML_CUDA=ON -DGGML_USE_PPU=ON -DGGML_NCP_QUACTLIZE=ON \
    -DGGML_NCP_FA=OFF -DGGML_NCP_MOE=OFF -DGGML_NCP_GDN=OFF \
    -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
    2>&1 | tee "$RUN/results/configure.log"
stage=build
TARGETS=(test-quactlize-ready llama-completion)
if [[ -n ${EVAL_FILE:-} ]]; then TARGETS+=(llama-perplexity); fi
cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j "$JOBS" \
    2>&1 | tee "$RUN/results/build.log"

stage=readiness
printf '\nKPACK_READY_PREFLIGHT\n'
timeout 60s "$BUILD_DIR/bin/test-quactlize-ready" 2>&1 | tee "$RUN/results/readiness.log"

ARGS=(-m "$MODEL" -ngl 99 --split-mode none --fit off
    -c 2048 -b 512 -ub 128 -t 16 -tb 32 --temp 0 --seed 0 -n 64
    --no-conversation --no-display-prompt --color off
    -p 'Explain why the sky is blue in two sentences.'
    -ot 'ffn_.*_exps=CUDA0_KPACK' -v)
for phase in baseline cold hit; do
    stage=$phase
    EXTRA=()
    if [[ $phase != baseline ]]; then EXTRA=(--kpack-cache "$RUN/cache"); fi
    printf '\nKPACK_MODEL_RUN phase=%s\n' "$phase"
    /usr/bin/time -v -o "$RUN/results/$phase.time" \
        "$BUILD_DIR/bin/llama-completion" "${ARGS[@]}" "${EXTRA[@]}" </dev/null \
        2>&1 | tee "$RUN/results/$phase.log"
    grep -q 'so-quactlize-kpack' "$RUN/results/$phase.log"
done

stage=check-cache
[[ -s $RUN/cache/manifest.json ]]
grep -q 'GPU pack queued' "$RUN/results/cold.log"
grep -q 'background write started:' "$RUN/results/cold.log"
grep -q 'slots_per_device=2 content_checks=disabled' "$RUN/results/cold.log"
grep -q 'published: total_seconds=' "$RUN/results/cold.log"
grep -q 'ready: tensors=.* content_checks=disabled' "$RUN/results/hit.log"
grep -qE 'cache_uploads=[1-9][0-9]* resident_misses=0' "$RUN/results/hit.log"
if grep -q 'GPU pack queued' "$RUN/results/hit.log"; then
    printf 'FAIL: cache-hit run repacked weights\n' >&2
    false
fi
cp "$RUN/cache/manifest.json" "$RUN/results/cache-manifest.json"
for phase in baseline cold hit; do
    {
        printf '\nKPACK_TIMING phase=%s\n' "$phase"
        grep -E 'Elapsed \(wall clock\)|User time|System time' "$RUN/results/$phase.time"
        grep -E 'llama_perf_context_print:|\[kpack-cache\]' "$RUN/results/$phase.log"
    } >> "$RUN/results/timing-summary.log"
done

if [[ -n ${EVAL_FILE:-} ]]; then
    # Use the existing GPU reference route, not a CPU model or CPU packer.
    # No-cache vs cache isolates persistence; ordinary weights vs K-pack measures
    # the compute-route difference. PPL/KLD are diagnostics, not an automatic
    # numerical admission threshold. Log-probability payloads stay on the box.
    EVAL_ARGS=(-m "$MODEL" -ngl 99 --split-mode none --fit off
        -c 256 -b "$EVAL_BATCH" -ub "$EVAL_BATCH" -t 16 -tb 32
        --chunks 2 --color off -f "$EVAL_FILE" -v)
    for phase in kpack-eval cache-eval reference-eval kpack-reference-eval; do
        stage=$phase
        case "$phase" in
            kpack-eval) EXTRA=(-ot 'ffn_.*_exps=CUDA0_KPACK' --save-all-logits "$RUN/kpack.logprobs");;
            cache-eval) EXTRA=(-ot 'ffn_.*_exps=CUDA0_KPACK' --kpack-cache "$RUN/cache"
                --kl-divergence --kl-divergence-base "$RUN/kpack.logprobs");;
            reference-eval) EXTRA=(-ot 'ffn_.*_exps=CUDA0' --save-all-logits "$RUN/reference.logprobs");;
            kpack-reference-eval) EXTRA=(-ot 'ffn_.*_exps=CUDA0_KPACK' --kpack-cache "$RUN/cache"
                --kl-divergence --kl-divergence-base "$RUN/reference.logprobs");;
        esac
        printf '\nKPACK_NUMERICAL_RUN phase=%s batch=%s\n' "$phase" "$EVAL_BATCH"
        /usr/bin/time -v -o "$RUN/results/$phase.time" \
            "$BUILD_DIR/bin/llama-perplexity" "${EVAL_ARGS[@]}" "${EXTRA[@]}" </dev/null \
            2>&1 | tee "$RUN/results/$phase.log"
        if [[ $phase == reference-eval ]]; then
            if grep -q 'so-quactlize-kpack' "$RUN/results/$phase.log"; then false; fi
        else
            grep -q 'so-quactlize-kpack' "$RUN/results/$phase.log"
        fi
        if [[ $phase == cache-eval || $phase == kpack-reference-eval ]]; then
            grep -q 'Mean    KLD:' "$RUN/results/$phase.log"
            grep -qE 'cache_uploads=[1-9][0-9]* resident_misses=0' "$RUN/results/$phase.log"
            if grep -q 'GPU pack queued' "$RUN/results/$phase.log"; then false; fi
        else
            grep -q 'Final estimate: PPL = ' "$RUN/results/$phase.log"
        fi
        if grep -qiE '(Final estimate: PPL =|Mean +KLD:|Maximum KLD:|Mean PPL\(Q\)/PPL\(base\)).*(nan|inf)' "$RUN/results/$phase.log"; then
            printf 'FAIL: nonfinite numerical metric\n' >&2
            false
        fi
        {
            printf '\nKPACK_NUMERICAL phase=%s batch=%s\n' "$phase" "$EVAL_BATCH"
            grep -E 'Final estimate:|Mean PPL|KLD:|Same top|llama_perf_context_print:' "$RUN/results/$phase.log"
        } >> "$RUN/results/numerical-summary.log"
    done
    printf 'KPACK_NUMERICAL_COMPARISON COMPLETE admission=PENDING_REVIEW\n' | tee -a "$RUN/results/numerical-summary.log"
else
    printf 'KPACK_NUMERICAL_COMPARISON NOT_RUN reason=EVAL_FILE_NOT_SET\n' > "$RUN/results/numerical-summary.log"
fi
stage=done
printf 'KPACK_MODEL_CACHE_SMOKE PASS\n' | tee "$RUN/results/verdict.log"
