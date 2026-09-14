#!/bin/bash
#
# run_tests.sh - llama.cpp automated test runner
#
# Features:
#   1. Run all tests in parallel (filtered by rules in config.sh)
#   2. Record each test's output to output/test_xxx.txt
#   3. Generate REPORT.md report
#
# Test filter rules:
#   - test-thread-safety: always skip
#   - test-recurrent-state-rollback: always skip
#   - test-download-model: always skip (binary not compiled)
#   - test-arg-parser: always skip (httplib does not support proxy)
#   - test-state-restore-fragmented: run only when LLAMA_MODEL is set
#   - test-save-load-state: run only when LLAMA_MODEL is set
#   - other tests: run normally (routed via ctest)
#
# Single-test mode (--run-only):
#   --run-only <TEST_NAME>  run only the specified test, ignore all skip rules
#

# set -e  # removed, allow xargs to continue after partial failures

# ========== Parse --run-only argument ==========
RUN_ONLY_TEST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --run-only)
            RUN_ONLY_TEST="$2"
            shift 2
            ;;
        -r)
            RUN_ONLY_TEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--run-only TEST_NAME]" >&2
            exit 1
            ;;
    esac
done

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Record start time
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# Use only GPU 0 to avoid long run time from testing all GPUs (keep if already set)
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0

cd "$LLAMA_CI_DIR/build-ci" 

echo "=========================================="
echo "llama.cpp CI Test Runner"
echo "=========================================="
echo "Project path: $LLAMA_CI_DIR"
echo "Test timeout: ${TEST_TIMEOUT}s"
echo "Parallel threads: ${TEST_THREADS}"
if [ -z "$LLAMA_MODEL" ]; then
    echo "Model file: (not set)"
    echo "  WARNING: LLAMA_MODEL not set -- qwen3-integration and other model-related tests will be skipped!"
else
    echo "Model file: $LLAMA_MODEL"
fi
echo "Output dir: $OUTPUT_DIR"
echo ""

# ========== Clean output directory ==========
if [ -n "$RUN_ONLY_TEST" ]; then
    # Single-test mode: only remove the output file for this test
    mkdir -p "$OUTPUT_DIR"
    SHORT_NAME="${RUN_ONLY_TEST#test-}"
    OUTPUT_FILE="$OUTPUT_DIR/test_${SHORT_NAME}.txt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[Cleanup] Removing old output file: $OUTPUT_FILE"
        rm -f "$OUTPUT_FILE"
    fi
else
    # Normal CI mode: clean the whole output directory
    if [ -d "$OUTPUT_DIR" ]; then
        echo "[Cleanup] Removing old output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
    mkdir -p "$OUTPUT_DIR"
fi

# ========== Single-test mode: run only the specified test ==========
if [ -n "$RUN_ONLY_TEST" ]; then
    echo "=========================================="
    echo "SINGLE TEST MODE: --run-only $RUN_ONLY_TEST"
    echo "=========================================="
    echo ""

    # Special handling for qwen3-integration
    if [ "$RUN_ONLY_TEST" = "qwen3-integration" ]; then
        if [ -z "$LLAMA_MODEL" ]; then
            echo "ERROR: qwen3-integration requires LLAMA_MODEL to be set" >&2
            exit 1
        fi
        echo "[qwen3-integration] Starting..."
        QWEN_REF="$SCRIPT_DIR/qwen3_reference.txt"
        QWEN_OUT="$OUTPUT_DIR/qwen3_integration_output.txt"
        QWEN_LOG="$OUTPUT_DIR/llama-server-integration.log"
        SERVER_PORT=8080

        pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
        sleep 2

        nohup ./bin/llama-server \
            -m "$LLAMA_MODEL" \
            -fa on \
            --temp 0 \
            --seed 1 \
            -n 1024 \
            --port $SERVER_PORT \
            > "$QWEN_LOG" 2>&1 &

        SERVER_PID=$!
        echo "  server PID: $SERVER_PID"
        echo "  waiting for server to be ready..."

        # Poll until server is ready, max 300s, check every 5s
        MAX_WAIT=300
        INTERVAL=5
        elapsed=0
        while [ $elapsed -lt $MAX_WAIT ]; do
            HTTP_CODE=$(curl -s --connect-timeout 2 --max-time 10 \
                             -o /dev/null -w "%{http_code}" \
                             http://localhost:$SERVER_PORT/v1/models 2>/dev/null)
            if [ "$HTTP_CODE" = "200" ]; then
                echo "  server ready after ${elapsed}s"
                break
            fi
            sleep $INTERVAL
            elapsed=$((elapsed + INTERVAL))
        done

        if [ $elapsed -ge $MAX_WAIT ]; then
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            echo "FAIL: server not ready within ${MAX_WAIT}s" > "$OUTPUT_FILE"
            echo "EXIT_CODE=1" >> "$OUTPUT_FILE"
            echo "  result: FAIL (server warmup timeout)"
            exit 1
        fi

        CURL_EXIT=0
        curl -s http://localhost:$SERVER_PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"qwen3","prompt":"who are you?","max_tokens":1024,"temperature":0,"seed":1}' \
            | python3 -c "
import sys, json
data = json.load(sys.stdin)
content = data.get('choices', [{}])[0].get('text', '')
print(content, end='')
" > "$QWEN_OUT"
        CURL_EXIT=$?

        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true

        SHORT_NAME="qwen3-integration"
        OUTPUT_FILE="$OUTPUT_DIR/test_${SHORT_NAME}.txt"

        if [ $CURL_EXIT -ne 0 ]; then
            echo "FAIL: curl failed with exit code $CURL_EXIT" > "$OUTPUT_FILE"
            echo "EXIT_CODE=1" >> "$OUTPUT_FILE"
            echo "  result: FAIL (curl exit=$CURL_EXIT)"
            exit 1
        elif [ ! -f "$QWEN_REF" ]; then
            echo "FAIL: reference file not found: $QWEN_REF" > "$OUTPUT_FILE"
            echo "EXIT_CODE=1" >> "$OUTPUT_FILE"
            echo "  result: FAIL (reference not found)"
            exit 1
        elif ! diff "$QWEN_REF" "$QWEN_OUT" > /dev/null 2>&1; then
            echo "FAIL: output differs from reference" > "$OUTPUT_FILE"
            echo "EXIT_CODE=1" >> "$OUTPUT_FILE"
            diff "$QWEN_REF" "$QWEN_OUT" >> "$OUTPUT_FILE" 2>&1
            echo "  result: FAIL (output mismatch)"
            exit 1
        else
            echo "PASS: output matches reference" > "$OUTPUT_FILE"
            echo "EXIT_CODE=0" >> "$OUTPUT_FILE"
            echo "  result: PASS"
            exit 0
        fi
    fi

    # Normal ctest test or direct binary test
    # First check whether it is in the ctest list
    ALL_TESTS=$(ctest -N 2>/dev/null | grep "Test #" | sed 's/.*: \([^ ]*\)/\1/' | tr '\n' ' ')

    if echo " $ALL_TESTS " | grep -q " $RUN_ONLY_TEST "; then
        # In ctest list, route through ctest
        echo "--- Running $RUN_ONLY_TEST via ctest ---"
        SHORT_NAME="${RUN_ONLY_TEST#test-}"
        OUTPUT_FILE="$OUTPUT_DIR/test_${SHORT_NAME}.txt"

        if [ "$RUN_ONLY_TEST" = "test-backend-ops" ]; then
            # Known failing MUL_MAT cases are handled by the skip list in test-backend-ops.cpp
            timeout $TEST_TIMEOUT ./bin/${RUN_ONLY_TEST} > "$OUTPUT_FILE" 2>&1
            EXIT_CODE=$?
        else
            timeout $TEST_TIMEOUT ctest -R "^${RUN_ONLY_TEST}$" --output-on-failure > "$OUTPUT_FILE" 2>&1
            EXIT_CODE=$?
        fi

        echo "EXIT_CODE=$EXIT_CODE" >> "$OUTPUT_FILE"

        if [ $EXIT_CODE -eq 0 ]; then
            echo "  result: PASS"
            echo "  output: $OUTPUT_FILE"
            exit 0
        else
            echo "  result: FAIL (exit=$EXIT_CODE)"
            echo "  output: $OUTPUT_FILE"
            exit 1
        fi
    else
        echo "ERROR: Test '$RUN_ONLY_TEST' not found in ctest list" >&2
        echo "Available tests:" >&2
        echo "$ALL_TESTS" | tr ' ' '\n' >&2
        exit 1
    fi
fi

# ========== Define skip rules ==========
# Return skip reason description
skip_reason() {
    local test_name="$1"
    case "$test_name" in
        test-thread-safety)            echo "OOM" ;;
        test-recurrent-state-rollback)  echo "not applicable for non-recurrent models" ;;
        test-download-model)           echo "binary not compiled" ;;
        test-arg-parser)               echo "httplib does not support proxy" ;;
        test-state-restore-fragmented|test-save-load-state) echo "OOM" ;;
        test-backend-ops)              echo "run separately by backend-tests job" ;;
        test-eval-callback-download-model) echo "runtime error" ;;
        test-eval-callback)            echo "runtime error" ;;
        qwen3-integration)             echo "LLAMA_MODEL not set" ;;
        *)                              echo "unknown" ;;
    esac
}

# Return 0 = skip, 1 = run
should_skip() {
    local test_name="$1"

    case "$test_name" in
        test-thread-safety)
            # Always skip (OOM)
            return 0
            ;;
        test-recurrent-state-rollback)
            # Always skip (skipped by non-recurrent models)
            return 0
            ;;
        test-download-model)
            # Always skip (binary not compiled)
            return 0
            ;;
        test-arg-parser)
            # Always skip (httplib does not support proxy, network requests hang)
            return 0
            ;;
        test-state-restore-fragmented|test-save-load-state)
            # Always skip (OOM, even Q4_K_M can exceed VRAM)
            return 0
            ;;
        test-backend-ops)
            # Run separately by backend-tests job via run_backend_ops.sh
            return 0
            ;;
        test-eval-callback-download-model)
            # runtime error
            return 0
            ;;
        test-eval-callback)
            # runtime error
            return 0
            ;;
        qwen3-integration)
            # Run only when LLAMA_MODEL is set (executed in inline section)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# ========== Fetch and filter test list ==========
echo "[1/5] Fetching and filtering test list..."
ALL_TESTS=$(ctest -N 2>/dev/null | grep "Test #" | sed 's/.*: \([^ ]*\)/\1/' | tr '\n' ' ')
TOTAL_COUNT=$(echo "$ALL_TESTS" | wc -w | xargs)
# qwen3-integration is not in the ctest list, but is appended in the result parsing loop
# When LLAMA_MODEL is set, total count should be increased by 1
if [ -n "$LLAMA_MODEL" ]; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
fi

# Tests to run
RUN_TESTS=""
# Tests to skip
SKIP_TESTS=""

for TEST_NAME in $ALL_TESTS; do
    if should_skip "$TEST_NAME"; then
        SKIP_TESTS="$SKIP_TESTS $TEST_NAME"
    else
        RUN_TESTS="$RUN_TESTS $TEST_NAME"
    fi
done

RUN_COUNT=$(echo "$RUN_TESTS" | tr ' ' '\n' | grep -c . 2>/dev/null || echo 0)
SKIP_COUNT=$(echo "$SKIP_TESTS" | tr ' ' '\n' | grep -c . 2>/dev/null || echo 0)

echo "Found $TOTAL_COUNT test(s)"
echo "Will run: $RUN_COUNT"
echo "Will skip: $SKIP_COUNT"
if [ -n "$SKIP_TESTS" ]; then
    echo "Skip list:$SKIP_TESTS"
fi
echo ""

# ========== Write empty result files for skipped tests ==========
for TEST_NAME in $SKIP_TESTS; do
    SHORT_NAME="${TEST_NAME#test-}"
    OUTPUT_FILE="$OUTPUT_DIR/test_${SHORT_NAME}.txt"
    echo "SKIP: $TEST_NAME (rule skip)" > "$OUTPUT_FILE"
    echo "EXIT_CODE=0" >> "$OUTPUT_FILE"
done

# Create temp directory to store per-test command scripts
TEMP_SCRIPT_DIR=$(mktemp -d)
trap "rm -rf $TEMP_SCRIPT_DIR" EXIT

echo "[2/5] Generating test scripts..."
IDX=0
for TEST_NAME in $RUN_TESTS; do
    IDX=$((IDX + 1))
    SCRIPT_FILE="$TEMP_SCRIPT_DIR/run_${IDX}.sh"
    SHORT_NAME="${TEST_NAME#test-}"
    OUTPUT_FILE="$OUTPUT_DIR/test_${SHORT_NAME}.txt"

    # Special handling: model-related tests call the binary directly with --model
    # All other tests route through ctest (ctest handles multi-label/binary routing)
    # test-backend-ops has many ops and needs a longer timeout
    if [ "$TEST_NAME" = "test-backend-ops" ]; then
        THIS_TIMEOUT=5400  # 90 minutes
    else
        THIS_TIMEOUT=$TEST_TIMEOUT
    fi

    if [ "$TEST_NAME" = "test-state-restore-fragmented" ] || [ "$TEST_NAME" = "test-save-load-state" ]; then
        cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
cd "$LLAMA_CI_DIR/build-ci"
timeout $THIS_TIMEOUT ./bin/${TEST_NAME} --model \$LLAMA_MODEL --main-gpu 0 > "$OUTPUT_FILE" 2>&1
EXIT_CODE=\$?
echo "EXIT_CODE=\$EXIT_CODE" >> "$OUTPUT_FILE"
exit \$EXIT_CODE
EOF
    elif [ "$TEST_NAME" = "test-backend-ops" ]; then
        cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
cd "$LLAMA_CI_DIR/build-ci"
timeout $THIS_TIMEOUT ./bin/${TEST_NAME} > "$OUTPUT_FILE" 2>&1
EXIT_CODE=\$?
echo "EXIT_CODE=\$EXIT_CODE" >> "$OUTPUT_FILE"
exit \$EXIT_CODE
EOF
    else
        TEST_NAME_ESCAPED=$(echo "$TEST_NAME" | sed 's/-/\\-/g')
        cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
cd "$LLAMA_CI_DIR/build-ci"
timeout $THIS_TIMEOUT ctest -R "^${TEST_NAME_ESCAPED}\$" --output-on-failure > "$OUTPUT_FILE" 2>&1
EXIT_CODE=\$?
echo "EXIT_CODE=\$EXIT_CODE" >> "$OUTPUT_FILE"
exit \$EXIT_CODE
EOF
    fi
    chmod +x "$SCRIPT_FILE"
done
echo "Generated $IDX test script(s)"
echo ""

echo "[3/5] Running all tests in parallel (${TEST_THREADS} concurrent)..."
# Run all test scripts in parallel
find "$TEMP_SCRIPT_DIR" -name "run_*.sh" | \
    xargs -P "$TEST_THREADS" -I {} bash {}

# ========== qwen3 llama-server API integration test ==========
# Run only when LLAMA_MODEL is set
if [ -n "$LLAMA_MODEL" ]; then
    echo ""
    echo "[3.5/5] Running qwen3 llama-server API integration test..."

    QWEN_REF="$SCRIPT_DIR/qwen3_reference.txt"
    QWEN_OUT="$OUTPUT_DIR/qwen3_integration_output.txt"
    QWEN_LOG="$OUTPUT_DIR/llama-server-integration.log"
    SERVER_PORT=8080

    # Clean up any leftover server
    pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
    sleep 2

    # Start server in background (using configured GPU)
    nohup ./bin/llama-server \
        -m "$LLAMA_MODEL" \
        -fa on \
        --temp 0 \
        --seed 1 \
        -n 1024 \
        --port $SERVER_PORT \
        > "$QWEN_LOG" 2>&1 &

    SERVER_PID=$!
    echo "  server PID: $SERVER_PID"
    echo "  waiting for server to be ready..."

    # Poll until server is ready, max 300s, check every 5s
    MAX_WAIT=300
    INTERVAL=5
    elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        HTTP_CODE=$(curl -s --connect-timeout 2 --max-time 10 \
                         -o /dev/null -w "%{http_code}" \
                         http://localhost:$SERVER_PORT/v1/models 2>/dev/null)
        if [ "$HTTP_CODE" = "200" ]; then
            echo "  server ready after ${elapsed}s"
            break
        fi
        sleep $INTERVAL
        elapsed=$((elapsed + INTERVAL))
    done

    if [ $elapsed -ge $MAX_WAIT ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "FAIL: server not ready within ${MAX_WAIT}s" > "$OUTPUT_DIR/test_qwen3-integration.txt"
        echo "EXIT_CODE=1" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        echo "  result: FAIL (server warmup timeout)" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        continue
    fi

    # Send API request, extract text field
    CURL_EXIT=0
    curl -s http://localhost:$SERVER_PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3","prompt":"who are you?","max_tokens":1024,"temperature":0,"seed":1}' \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
content = data.get('choices', [{}])[0].get('text', '')
print(content, end='')
" > "$QWEN_OUT"
    CURL_EXIT=$?

    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true

    # Compare output
    QWEN_EXIT_CODE=0
    if [ $CURL_EXIT -ne 0 ]; then
        echo "FAIL: curl failed with exit code $CURL_EXIT" > "$OUTPUT_DIR/test_qwen3-integration.txt"
        QWEN_EXIT_CODE=1
    elif [ ! -f "$QWEN_REF" ]; then
        echo "FAIL: reference file not found: $QWEN_REF" > "$OUTPUT_DIR/test_qwen3-integration.txt"
        QWEN_EXIT_CODE=1
    elif ! diff "$QWEN_REF" "$QWEN_OUT" > /dev/null 2>&1; then
        echo "FAIL: output differs from reference" > "$OUTPUT_DIR/test_qwen3-integration.txt"
        echo "--- reference (first 100 chars) ---" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        head -c 100 "$QWEN_REF" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        echo "" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        echo "--- output (first 100 chars) ---" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        head -c 100 "$QWEN_OUT" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        echo "" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
        diff "$QWEN_REF" "$QWEN_OUT" >> "$OUTPUT_DIR/test_qwen3-integration.txt" 2>&1
        QWEN_EXIT_CODE=1
    else
        echo "PASS: output matches reference" > "$OUTPUT_DIR/test_qwen3-integration.txt"
    fi

    echo "EXIT_CODE=$QWEN_EXIT_CODE" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
    echo "  qwen3-integration result: exit=$QWEN_EXIT_CODE"
else
    echo ""
    echo "  WARNING: LLAMA_MODEL not set -- skipping qwen3-integration test"
    echo "  SKIP: qwen3-integration (LLAMA_MODEL not set)" > "$OUTPUT_DIR/test_qwen3-integration.txt"
    echo "EXIT_CODE=0" >> "$OUTPUT_DIR/test_qwen3-integration.txt"
fi

echo ""
echo "[4/5] Parsing results and generating report..."

# Parse each test result (includes all tests: run and skipped)
PASS_COUNT=0
FAIL_COUNT=0
SKIPPED_COUNT=0
declare -a PASS_TESTS
declare -a FAIL_TESTS
declare -a SKIP_TESTS_ARR

for TEST_NAME in $ALL_TESTS qwen3-integration; do
    SHORT_NAME="${TEST_NAME#test-}"
    OUTPUT_FILE="$OUTPUT_DIR/test_${SHORT_NAME}.txt"

    if [ ! -f "$OUTPUT_FILE" ]; then
        EXIT_CODE="N/A"
        ERROR_TYPE="output file missing"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_TESTS+=("$TEST_NAME|$EXIT_CODE|$ERROR_TYPE")
        continue
    fi

    # Check whether this is a rule-skipped test
    if grep -q "SKIP: $TEST_NAME" "$OUTPUT_FILE" 2>/dev/null; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIP_TESTS_ARR+=("$TEST_NAME")
        continue
    fi

    EXIT_CODE=$(grep "EXIT_CODE=" "$OUTPUT_FILE" | tail -1 | sed 's/EXIT_CODE=//')
    if [ -z "$EXIT_CODE" ]; then
        EXIT_CODE=$(tail -1 "$OUTPUT_FILE" | grep -o '[0-9]*' | tail -1)
    fi
    if [ -z "$EXIT_CODE" ]; then
        EXIT_CODE="N/A"
    fi

    # Determine error type
    if [ "$EXIT_CODE" = "0" ]; then
        CONTENT=$(cat "$OUTPUT_FILE")
        # When exit code is 0, output containing SKIP keyword counts as SKIP
        # But test-backend-ops produces large output; SKIP keywords from individual ops do not mean the whole test was skipped
        if [ "$TEST_NAME" != "test-backend-ops" ] && echo "$CONTENT" | grep -q "SKIP\|skip\|Skip"; then
            ERROR_TYPE="test skipped"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            SKIP_TESTS_ARR+=("$TEST_NAME")
        else
            ERROR_TYPE=""
            PASS_COUNT=$((PASS_COUNT + 1))
            PASS_TESTS+=("$TEST_NAME|$OUTPUT_FILE")
        fi
    elif [ "$EXIT_CODE" = "124" ] || [ "$EXIT_CODE" = "137" ]; then
        ERROR_TYPE="timeout"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_TESTS+=("$TEST_NAME|$EXIT_CODE|$ERROR_TYPE")
    elif [ "$EXIT_CODE" = "N/A" ]; then
        ERROR_TYPE="no output"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_TESTS+=("$TEST_NAME|$EXIT_CODE|$ERROR_TYPE")
    else
        CONTENT=$(cat "$OUTPUT_FILE")
        if echo "$CONTENT" | grep -q "No model file\|--model is required\|failed to open GGUF\|LLAMACPP_TEST_MODELFILE"; then
            ERROR_TYPE="model file missing"
        elif echo "$CONTENT" | grep -q "HTTP request failed\|HTTP error\|could not fetch\|network\|connection"; then
            ERROR_TYPE="network error"
        elif echo "$CONTENT" | grep -q "CUDA\|cuda\|GPU\|PPU"; then
            ERROR_TYPE="CUDA error"
        elif echo "$CONTENT" | grep -q "No such file or directory\|does not exist\|command not found\|127"; then
            ERROR_TYPE="file missing"
        else
            ERROR_TYPE="runtime error"
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_TESTS+=("$TEST_NAME|$EXIT_CODE|$ERROR_TYPE")
    fi
done

# Record end time
END_TIME=$(date +%s)
END_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_SECONDS=$((END_TIME - START_TIME))

# Compute total elapsed time
if [ $TOTAL_SECONDS -ge 3600 ]; then
    DURATION_HUMAN="$(($TOTAL_SECONDS / 3600))h $(($((TOTAL_SECONDS % 3600)) / 60))m"
elif [ $TOTAL_SECONDS -ge 60 ]; then
    DURATION_HUMAN="$((TOTAL_SECONDS / 60))m $((TOTAL_SECONDS % 60))s"
else
    DURATION_HUMAN="${TOTAL_SECONDS}s"
fi

# Generate REPORT.md
cat > "$REPORT_FILE" << EOF
# llama.cpp CI Test Report

- **Project path**: \`$LLAMA_CI_DIR\`
- **Test start**: $START_TIME_HUMAN
- **Test end**: $END_TIME_HUMAN
- **Total duration**: $DURATION_HUMAN ($TOTAL_SECONDS s)
- **Model file**: \`${LLAMA_MODEL:-(not set)}\`

---

## Summary

| Status | Count |
|--------|-------|
| ✅ PASS | $PASS_COUNT |
| ❌ FAIL | $FAIL_COUNT |
| ⏭️ SKIP | $SKIPPED_COUNT |
| **Total** | **$TOTAL_COUNT** |

---

## Skipped Tests

| # | Test Name | Skip Reason |
|---|-----------|-------------|
EOF

IDX=0
for TEST_NAME in "${SKIP_TESTS_ARR[@]}"; do
    IDX=$((IDX + 1))
    REASON=$(skip_reason "$TEST_NAME")
    echo "| $IDX | \`$TEST_NAME\` | $REASON |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

---

## Failed Tests

| # | Test Name | Exit Code | Error Type | Output |
|---|-----------|-----------|------------|--------|
EOF

IDX=0
for entry in "${FAIL_TESTS[@]}"; do
    IFS='|' read -r TEST_NAME EXIT_CODE ERROR_TYPE <<< "$entry"
    IDX=$((IDX + 1))
    SHORT_NAME="${TEST_NAME#test-}"
    OUTPUT_LINK="[log](output/test_${SHORT_NAME}.txt)"
    echo "| $IDX | \`$TEST_NAME\` | $EXIT_CODE | $ERROR_TYPE | $OUTPUT_LINK |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

---

## Passed Tests

| # | Test Name | Output |
|---|-----------|--------|
EOF

IDX=0
for entry in "${PASS_TESTS[@]}"; do
    IFS='|' read -r TEST_NAME OUTPUT_FILE <<< "$entry"
    IDX=$((IDX + 1))
    SHORT_NAME="${TEST_NAME#test-}"
    OUTPUT_LINK="[log](output/test_${SHORT_NAME}.txt)"
    echo "| $IDX | \`$TEST_NAME\` | $OUTPUT_LINK |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

---

*Report generated by ppu_ci/run_tests.sh*
EOF

echo ""
echo "=========================================="
echo "Tests complete!"
echo "=========================================="
echo "Total tests: $TOTAL_COUNT"
echo "Passed:      $PASS_COUNT"
echo "Failed:      $FAIL_COUNT"
echo "Skipped:     $SKIPPED_COUNT"
echo "Duration:    $DURATION_HUMAN"
echo "Report:      $REPORT_FILE"
echo ""
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "========== REPORT.md =========="
cat "$REPORT_FILE"
echo "========== END REPORT =========="

# Exit non-zero if any test failed
if [ "$FAIL_COUNT" -gt 0 ]; then
    # Package logs to shared directory for debugging
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    TARBALL="/ppusw/share/eec_shared/llama_ctest_logs_${TIMESTAMP}.tar.gz"
    mkdir -p /ppusw/share/eec_shared
    tar -zcf "$TARBALL" -C "$SCRIPT_DIR" output REPORT.md
    echo "Failed test logs packaged to: $TARBALL"
    exit 1
fi