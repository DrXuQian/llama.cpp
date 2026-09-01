#!/usr/bin/env bash

set -euo pipefail

# llama.cpp/tools/batched-bench/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_PATH="${REPO_ROOT}/build/bin/llama-batched-bench"


DEFAULT_DEVICE="7"                          # CUDA_VISIBLE_DEVICES
DEFAULT_NPPS="5120,4096,2048,1024,512"      # -npp
DEFAULT_NTGS="2048,1024,512"                # -ntg
DEFAULT_NPL="1"                             # -npl
DEFAULT_NGL="all"                           # -ngl  number | auto | all
DEFAULT_BATCH="5120"                        # -b
DEFAULT_UBATCH="5120"                       # -ub
DEFAULT_EXTRA_FLAGS="-fa 1"                 


declare -A MODEL_CUSTOM_CONFIGS
MODEL_CUSTOM_CONFIGS["Qwen3.5-122B-A10B-Q4_K_M"]="CUDA_DEVICES=6,7 EXTRA=\"-sm tensor -ts 1,1\""
# MODEL_CUSTOM_CONFIGS["Qwen3-32B-Q4_K_M"]="CUDA_DEVICES=0,1"

# get model tag
model_tag() {
    local base
    base="$(basename "$1")"
    base="${base%.gguf}"
    base="${base%-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]}"
    echo "$base"
}

# ============================================================================
# check all model
# ============================================================================
SKIP_CONFIRM=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        -y|--yes) SKIP_CONFIRM=1 ;;
        *)         ARGS+=("$arg") ;;
    esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

[[ $# -ge 1 ]] || { echo "use: $0 [-y] <models_root>" >&2; exit 1; }
ROOT_DIR="$1"
[[ -d "$ROOT_DIR" ]] || { echo "ERROR: dir not found: $ROOT_DIR" >&2; exit 1; }
[[ -x "$BIN_PATH" ]] || { echo "ERROR: Executable file not found: $BIN_PATH" >&2; exit 1; }

echo "============================================================"
echo " llama-batched-bench"
echo " Root: $ROOT_DIR"
echo " Binary: $BIN_PATH | device: $DEFAULT_DEVICE | NGL: $DEFAULT_NGL"
echo " NPPS: $DEFAULT_NPPS | NTGS: $DEFAULT_NTGS | BATCH: $DEFAULT_BATCH | UB: $DEFAULT_UBATCH"
echo "============================================================"

# ============================================================================
# scan model
# ============================================================================
declare -A SELECTED
while IFS= read -r f; do
    bn="$(basename "$f")"
    if [[ "$bn" == *-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf ]] \
       && [[ "$bn" != *-00001-of-[0-9][0-9][0-9][0-9][0-9].gguf ]]; then
        continue
    fi
    SELECTED["$(model_tag "$f")"]="$f"
done < <(find "$ROOT_DIR" -type f -name '*.gguf')

MODELS=()
for _tag in "${!SELECTED[@]}"; do
    MODELS+=("${SELECTED[$_tag]}")
done
if [[ ${#MODELS[@]} -gt 0 ]]; then
    mapfile -t MODELS < <(printf '%s\n' "${MODELS[@]}" | sort)
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: not find any model files in $ROOT_DIR" >&2; exit 1
fi

echo
echo "------------------------------------------------------------"
echo " test ${#MODELS[@]} models："
i=0
for m in "${MODELS[@]}"; do
    i=$((i+1))
    tag="$(model_tag "$m")"
    cfg="${MODEL_CUSTOM_CONFIGS[$tag]:-}"
    if [[ -n "$cfg" ]]; then
        printf '   %2d) %s  [custom]\n' "$i" "$m"
    else
        printf '   %2d) %s\n' "$i" "$m"
    fi
done
echo "------------------------------------------------------------"

if [[ $SKIP_CONFIRM -eq 0 ]]; then
    echo
    read -r -p "Start test？[y/N] " confirm
    case "$confirm" in
        y|Y|yes|YES) echo "Start..." ;;
        *) echo "canceled，exit。"; exit 0 ;;
    esac
fi

# ============================================================================
# Start run
# ============================================================================
FAIL=0; TOTAL=0
for model in "${MODELS[@]}"; do
    TOTAL=$((TOTAL+1))

    echo
    echo "------------------------------------------------------------"
    echo ">> Testing: $model"

    CUDA_DEVICES="$DEFAULT_DEVICE"
    NPPS="$DEFAULT_NPPS"; NTGS="$DEFAULT_NTGS"; NPL="$DEFAULT_NPL"
    NGL="$DEFAULT_NGL"; BATCH="$DEFAULT_BATCH"; UBATCh="$DEFAULT_UBATCH"
    EXTRA=""
    tag="$(model_tag "$model")"
    cfg="${MODEL_CUSTOM_CONFIGS[$tag]:-}"
    [[ -n "$cfg" ]] && echo "  custom  ($tag): $cfg" && eval "$cfg"
    all_extra="$DEFAULT_EXTRA_FLAGS $EXTRA"

    printf '  $ CUDA_VISIBLE_DEVICES=%s %s -m "%s" -npp %s -ntg %s -npl %s -ngl %s -b %s -ub %s %s\n\n' \
        "$CUDA_DEVICES" "$BIN_PATH" "$model" "$NPPS" "$NTGS" "$NPL" "$NGL" "$BATCH" "$UBATCh" "$all_extra"

    if CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$BIN_PATH" \
        -m "$model" -npp "$NPPS" -ntg "$NTGS" -npl "$NPL" \
        -ngl "$NGL" -b "$BATCH" -ub "$UBATCh" $all_extra; then
        echo "  [OK]"
    else
        echo "  [FAIL] exit: $?" >&2; FAIL=$((FAIL+1))
    fi
done

# ============================================================================
# 汇总
# ============================================================================
echo
echo "============================================================"
echo " finish（Fail $FAIL / All $TOTAL）"
echo "============================================================"
[[ $FAIL -eq 0 ]]
