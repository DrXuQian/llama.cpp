#!/bin/bash
#
# run_backend_ops.sh - run test-backend-ops, optionally filtered to a subset of ops
#
# Usage: bash run_backend_ops.sh [op1,op2,op3]
#   No argument : run all ops
#   e.g.: bash run_backend_ops.sh MUL_MAT
#         bash run_backend_ops.sh MUL_MAT,COPY
#         bash run_backend_ops.sh FLASH_ATTN_EXT,MUL_MAT
#

set -e

OPS="${1:-}"   # empty = run all ops
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$LLAMA_CI_DIR" ]; then
    echo "Error: LLAMA_CI_DIR is not set"
    echo "Set the llama.cpp repository path first, e.g.:"
    echo "  export LLAMA_CI_DIR=/path/to/your/llama.cpp"
    exit 1
fi

if [ ! -d "$LLAMA_CI_DIR/build-ci" ]; then
    echo "Error: $LLAMA_CI_DIR/build-ci not found; make sure LLAMA_CI_DIR points to a built llama.cpp repository"
    exit 1
fi

export LLAMA_CI_DIR

OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

if [ -n "$OPS" ]; then
    FILTER_NAME="${OPS//,/_}"
    OUTPUT_FILE="$OUTPUT_DIR/test_backend_ops_${FILTER_NAME}.txt"
    echo "=========================================="
    echo "test-backend-ops (ops=$OPS)"
    echo "=========================================="
else
    OUTPUT_FILE="$OUTPUT_DIR/test_backend_ops_all.txt"
    echo "=========================================="
    echo "test-backend-ops (all ops)"
    echo "=========================================="
fi
echo "Output: $OUTPUT_FILE"

cd "$LLAMA_CI_DIR/build-ci"

# Use only GPU 0 to avoid long run time from testing all GPUs
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0

if [ -n "$OPS" ]; then
    timeout 5400 ./bin/test-backend-ops -o "$OPS" > "$OUTPUT_FILE" 2>&1
else
    timeout 14400 ./bin/test-backend-ops > "$OUTPUT_FILE" 2>&1
fi

EXIT_CODE=$?
echo "EXIT_CODE=$EXIT_CODE" >> "$OUTPUT_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "  -> PASS"
else
    echo "  -> FAIL (exit=$EXIT_CODE)"
fi

echo "========== OUTPUT =========="
echo "Done. Output: $OUTPUT_FILE"

# Package logs to shared directory if test failed
if [ "$EXIT_CODE" -ne 0 ]; then
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    TARBALL="/ppusw/share/eec_shared/llama_backend_ops_logs_${TIMESTAMP}.tar.gz"
    mkdir -p /ppusw/share/eec_shared
    tar -zcf "$TARBALL" -C "$SCRIPT_DIR" output
    echo "Failed test logs packaged to: $TARBALL"
    exit 1
fi
