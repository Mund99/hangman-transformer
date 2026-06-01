#!/usr/bin/env bash
# Third round of supervised search — map the convergence ceiling.
#
# All experiments use normal curriculum (4×/2×), no early-game bias.
# Building on exp_d (100k, 64.63% test) which was still climbing.
#
#   exp_g: 125k steps  — expect ~+1.2pp over exp_d
#   exp_h: 150k steps  — expect ~+0.8pp over exp_g
#   exp_i: 200k steps  — expect ~+0.5pp over exp_h (ceiling signal)
#
# Usage:
#   bash scripts/run_supervised_search3.sh
#   bash scripts/run_supervised_search3.sh &   # background (recommended)

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search3_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search3_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════"
_log "  Supervised search3 run: ${SEARCH_ID}"
_log "  All experiments: curriculum(4×/2×), no early-game bias"
_log "  exp_g: 125k steps"
_log "  exp_h: 150k steps"
_log "  exp_i: 200k steps"
_log "  baseline: exp_d 100k+curr(4×/2×) → val=62.15%  test=64.63%"
_log "═══════════════════════════════════════════════════════"

run_exp() {
    local name="$1"
    local steps="$2"

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}  (steps=${steps})"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    python scripts/train_supervised.py \
        --steps          "$steps" \
        --bs             512      \
        --lr             3e-4     \
        --d-model        256      \
        --n-heads        4        \
        --n-layers       4        \
        --dim-ff         1024     \
        --log-every      200      \
        --eval-every     2000     \
        --demo-every     1000     \
        --val-subset     2000     \
        --workers        4        \
        --length-curriculum       \
        --short-weight   4.0      \
        --medium-weight  2.0      \
        --ckpt-dir       "$ckpt_dir" \
        --log-file       "${exp_dir}/supervised.log"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    local val_pct
    val_pct=$(grep "saved checkpoints/supervised_best.pt" "${exp_dir}/supervised.log" \
              | tail -1 | grep -oP 'val=\K[0-9.]+' || echo "n/a")
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"
    _log "  best val: ${val_pct}%"

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

    LAST_VAL="$val_pct"
    LAST_TEST="$test_pct"
}

# ── Run experiments sequentially ─────────────────────────────────────────────

run_exp "exp_g_125k_curr_4x2x" 125000
VAL_G="$LAST_VAL"; TEST_G="$LAST_TEST"

run_exp "exp_h_150k_curr_4x2x" 150000
VAL_H="$LAST_VAL"; TEST_H="$LAST_TEST"

run_exp "exp_i_200k_curr_4x2x" 200000
VAL_I="$LAST_VAL"; TEST_I="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  SEARCH3 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════"
_log "  experiment                   val%       test%    gain over prev"
_log "  ──────────────────────────────────────────────────────────────"
_log "  baseline exp_d 100k(4×/2×)   62.15%     64.63%   —"
_log "  exp_g 125k curr(4×/2×)        ${VAL_G}%     ${TEST_G}%"
_log "  exp_h 150k curr(4×/2×)        ${VAL_H}%     ${TEST_H}%"
_log "  exp_i 200k curr(4×/2×)        ${VAL_I}%     ${TEST_I}%"
_log "  ──────────────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "═══════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
