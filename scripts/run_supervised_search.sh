#!/usr/bin/env bash
# Sequential supervised-only search across three experiment variants.
#
# Experiments (run one after another on the single GPU):
#   exp_a: 75k steps + length-curriculum + early-game-bias
#   exp_b: 50k steps + length-curriculum + early-game-bias
#   exp_c: 75k steps + length-curriculum only
#
# Each experiment gets its own subdirectory under logs/sup_search_YYYYMMDD_HHMM/:
#   exp_a_75k_curr_eb/   supervised.log, eval_test.log, checkpoints/
#   exp_b_50k_curr_eb/   …
#   exp_c_75k_curr/      …
#
# A summary.log is written at the end comparing val% and test% for all three.
#
# Usage:
#   bash scripts/run_supervised_search.sh
#   bash scripts/run_supervised_search.sh &   # background (recommended)

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

# latest symlink
ln -sfn "sup_search_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════"
_log "  Supervised search run: ${SEARCH_ID}"
_log "  exp_a: 75k + curriculum + early_game_bias"
_log "  exp_b: 50k + curriculum + early_game_bias"
_log "  exp_c: 75k + curriculum only"
_log "═══════════════════════════════════════════════════════"

# ── helper: run one experiment ──────────────────────────────────────────────
run_exp() {
    local name="$1"      # e.g. exp_a_75k_curr_eb
    local steps="$2"     # e.g. 75000
    local extra="$3"     # e.g. "--length-curriculum --early-game-bias"

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}  (steps=${steps})"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    # shellcheck disable=SC2086
    python scripts/train_supervised.py \
        --steps      "$steps" \
        --bs         512      \
        --lr         3e-4     \
        --d-model    256      \
        --n-heads    4        \
        --n-layers   4        \
        --dim-ff     1024     \
        --log-every  200      \
        --eval-every 2000     \
        --demo-every 1000     \
        --val-subset 2000     \
        --workers    4        \
        --ckpt-dir   "$ckpt_dir" \
        --log-file   "${exp_dir}/supervised.log" \
        $extra

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    # Extract best val win-rate from log
    local val_pct
    val_pct=$(grep "saved checkpoints/supervised_best.pt" "${exp_dir}/supervised.log" \
              | tail -1 | grep -oP 'val=\K[0-9.]+' || echo "n/a")
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"
    _log "  best val: ${val_pct}%"

    # Test evaluation
    local sup_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ -f "$sup_ckpt" ]]; then
        _log "  Running test evaluation…"
        python scripts/evaluate.py \
            --ckpt     "$sup_ckpt" \
            --words    data/test_words.txt \
            --log-file "${exp_dir}/eval_test.log"

        local test_pct
        test_pct=$(grep -oP 'TEST WIN RATE\s+:\s+\K[0-9.]+' "${exp_dir}/eval_test.log" \
                   | tail -1 || echo "n/a")
        echo "test_win_rate: ${test_pct}%" >> "${exp_dir}/run_info.txt"
        echo "eval_done: $(date)" >> "${exp_dir}/run_info.txt"
        _log "  test win-rate: ${test_pct}%"
    else
        _log "  WARNING: no checkpoint found at ${sup_ckpt}"
        test_pct="n/a"
    fi

    # Return values via global variables
    LAST_VAL="$val_pct"
    LAST_TEST="$test_pct"
}

# ── Run experiments sequentially ─────────────────────────────────────────────

run_exp "exp_a_75k_curr_eb" 75000 "--length-curriculum --early-game-bias"
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"

run_exp "exp_b_50k_curr_eb" 50000 "--length-curriculum --early-game-bias"
VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"

run_exp "exp_c_75k_curr" 75000 "--length-curriculum"
VAL_C="$LAST_VAL"; TEST_C="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  SEARCH COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════"
_log "  experiment              val%       test%"
_log "  ─────────────────────────────────────────────────"
_log "  exp_a 75k+curr+ebias    ${VAL_A}%     ${TEST_A}%"
_log "  exp_b 50k+curr+ebias    ${VAL_B}%     ${TEST_B}%"
_log "  exp_c 75k+curr          ${VAL_C}%     ${TEST_C}%"
_log "  ─────────────────────────────────────────────────"
_log "  baseline (run3, no curr) val≈58.80%  test≈59.84%"
_log "═══════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
