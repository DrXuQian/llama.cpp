#!/bin/bash
# Incremental clang-format check for AONE CI.
#
# Only checks lines changed in the current PR/MR, not entire files.
# Supports both apt-based (Ubuntu/Debian) and yum/dnf-based
# (RHEL/Alibaba Cloud Linux) distros.
#
# Environment variables used:
#   MR_TARGET   - target branch for merge requests (optional)
#   PUSH_BEFORE - previous commit SHA for push events (optional)

set -e

# Auto-detect package manager and install clang-format
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq clang-format-18 git python3
elif command -v yum >/dev/null 2>&1; then
    yum install -y clang-tools-extra git python3
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y clang-tools-extra git python3
else
    echo "❌ No supported package manager found" >&2
    exit 1
fi

# Determine diff base
MR_TARGET="${MR_TARGET:-}"
PUSH_BEFORE="${PUSH_BEFORE:-}"
if [ "$MR_TARGET" != "" ] && [ "$MR_TARGET" != "null" ]; then
    git fetch origin "$MR_TARGET" --depth=1 || true
    DIFF_BASE="origin/$MR_TARGET"
elif [ "$PUSH_BEFORE" != "" ] && [ "$PUSH_BEFORE" != "null" ]; then
    DIFF_BASE="$PUSH_BEFORE"
else
    DIFF_BASE="HEAD~1"
fi

# Ensure HEAD~1 is available in shallow checkouts (e.g., manual runs)
if [ "$DIFF_BASE" = "HEAD~1" ]; then
    if ! git rev-parse HEAD~1 >/dev/null 2>&1; then
        git fetch --depth=2 origin 2>/dev/null || true
    fi
fi

echo "=============================================="
echo "Diff base: $DIFF_BASE"
echo "=============================================="

# Get changed C++ files (early exit if none)
CHANGED=$(git diff "$DIFF_BASE"..HEAD --name-only --diff-filter=d -- '*.cpp' '*.h' '*.hpp' '*.c' 2>/dev/null || true)
if [ -z "$CHANGED" ]; then
    echo "✅ No C++ files changed, skipping clang-format"
    exit 0
fi
echo "=============================================="
echo "Changed C++ files:"
echo "$CHANGED"
echo "=============================================="

# Locate git-clang-format from the installed clang-format package
GIT_CF=$(command -v git-clang-format || true)
if [ -z "$GIT_CF" ]; then
    GIT_CF=$(find /usr/share/clang -name git-clang-format -print -quit 2>/dev/null || true)
fi
if [ -z "$GIT_CF" ] || [ ! -x "$GIT_CF" ]; then
    echo "❌ git-clang-format not found; ensure clang-format package is installed" >&2
    exit 1
fi
echo "Using git-clang-format: $GIT_CF"

# Locate the clang-format binary (name varies by distro/package)
CF_BIN=$(command -v clang-format-18 || command -v clang-format || true)
if [ -z "$CF_BIN" ]; then
    echo "❌ clang-format binary not found" >&2
    exit 1
fi
echo "Using clang-format binary: $CF_BIN"

# Check only the lines changed in this PR/MR
CF_STATUS=0
RAW_OUTPUT=$("$GIT_CF" --binary "$CF_BIN" --diff "$DIFF_BASE" -- '*.cpp' '*.h' '*.hpp' '*.c' 2>&1) || CF_STATUS=$?

echo "RAW_OUTPUT:"
echo "$RAW_OUTPUT"

# Filter out informational messages that are not actual formatting diffs
FORMAT_DIFF=$(echo "$RAW_OUTPUT" | grep -v '^clang-format did not modify any files$' || true)

# Some git-clang-format versions return non-zero when formatting issues exist
if [ "$CF_STATUS" -ne 0 ] && [ -z "$FORMAT_DIFF" ]; then
    echo "❌ git-clang-format failed to run:" >&2
    echo "$RAW_OUTPUT" >&2
    exit 1
fi

if [ -n "$FORMAT_DIFF" ]; then
    echo "❌ clang-format check failed (changed lines only):"
    echo "$FORMAT_DIFF"
    exit 1
fi
echo "✅ clang-format passed"
