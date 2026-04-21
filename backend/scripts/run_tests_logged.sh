#!/bin/bash
# Crash-survivable pytest runner.
#
# Every invocation creates a timestamped log dir with:
#   - progress.log: one line per test file (START / PASS / FAIL / EXIT=N)
#   - details.log:  full pytest output of each file
#   - gpu.log:      GPU memory snapshots before/after each file
#   - summary.txt:  at the end, pass/fail counts and failing file list
#
# Every log write is followed by `sync` so the disk is flushed immediately.
# If the system freezes/reboots mid-run, the logs up to the crash point survive.
#
# Usage (from backend/):
#   bash scripts/run_tests_logged.sh                     # all test files
#   bash scripts/run_tests_logged.sh tests/test_foo.py   # specific file(s)
#
# After a crash + reboot:
#   ls -t ~/.nous-test-runs/ | head -1       # most recent run
#   tail -5 ~/.nous-test-runs/<latest>/progress.log   # last file it was running
#   cat ~/.nous-test-runs/<latest>/details.log | less  # full output

set -u

TS=$(date +%Y-%m-%dT%H-%M-%S)
RUN_DIR="$HOME/.nous-test-runs/$TS"
mkdir -p "$RUN_DIR"

PROGRESS="$RUN_DIR/progress.log"
DETAILS="$RUN_DIR/details.log"
GPU_LOG="$RUN_DIR/gpu.log"
SUMMARY="$RUN_DIR/summary.txt"

# Record run metadata
{
    echo "=== Test run $TS ==="
    echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "Commit: $(git rev-parse --short HEAD 2>/dev/null)"
    echo "Dirty:  $([[ -n $(git status --porcelain 2>/dev/null) ]] && echo yes || echo no)"
    echo "CWD:    $(pwd)"
    echo "User:   $(whoami)"
    echo "Host:   $(hostname)"
    echo ""
} | tee "$PROGRESS"
sync

# Snapshot GPU state
snap_gpu() {
    local label="$1"
    {
        echo "$(date +%T) [$label]"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
        echo ""
    } >> "$GPU_LOG"
    sync
}

# Default: all tests/test_*.py. Or use arguments.
if [[ $# -eq 0 ]]; then
    FILES=($(ls tests/test_*.py 2>/dev/null | sort))
else
    FILES=("$@")
fi

echo "Running ${#FILES[@]} test files. Log dir: $RUN_DIR" | tee -a "$PROGRESS"
sync

PASS_COUNT=0
FAIL_COUNT=0
FAIL_FILES=()

snap_gpu "run-start"

for f in "${FILES[@]}"; do
    [[ ! -f "$f" ]] && continue
    snap_gpu "pre:$f"
    echo "$(date +%T) START $f" | tee -a "$PROGRESS"
    sync

    # 60s per-file timeout, CUDA hidden, bg tasks disabled, forked isolation
    if stdbuf -oL timeout --kill-after=5s 60 env \
            CUDA_VISIBLE_DEVICES="" \
            NOUS_DISABLE_BG_TASKS=1 \
            uv run pytest "$f" --tb=line -q -p no:cacheprovider --forked 2>&1 \
            | tee -a "$DETAILS"; then
        echo "$(date +%T) PASS $f" | tee -a "$PROGRESS"
        sync
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        rc=$?
        echo "$(date +%T) FAIL $f (exit=$rc)" | tee -a "$PROGRESS"
        sync
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_FILES+=("$f")
    fi
    snap_gpu "post:$f"
done

snap_gpu "run-end"

# Write summary
{
    echo "=== Summary ==="
    echo "Run dir: $RUN_DIR"
    echo "Total:   ${#FILES[@]}"
    echo "Passed:  $PASS_COUNT"
    echo "Failed:  $FAIL_COUNT"
    echo ""
    if [[ ${#FAIL_FILES[@]} -gt 0 ]]; then
        echo "Failed files:"
        for f in "${FAIL_FILES[@]}"; do
            echo "  - $f"
        done
    fi
    echo ""
    echo "Full details: $DETAILS"
    echo "GPU trace:    $GPU_LOG"
} | tee "$SUMMARY"
sync

echo ""
echo "Done. Log dir: $RUN_DIR"
