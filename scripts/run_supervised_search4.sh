#!/usr/bin/env bash
# Fourth round of supervised search — new expanded dataset (353k words, max_len=45).
#
# Dataset: NLTK corpus + competition word list (was 251k, now 353k words)
# Architecture unchanged: d_model=256, 4 heads, 4 layers, dim_ff=1024
# All experiments: curriculum(4×/2×), no early-game bias
#
# Previous best: exp_i  200k curr(4×/2×)  val=65.35%  test=68.12%  [251k dataset]
#
#   exp_j: 200k steps — new-data baseline (direct comparison)
#   exp_k: 250k steps — push further
#   exp_l: 300k steps — find new ceiling
#
# Usage:
#   bash scripts/run_supervised_search4.sh
#   bash scripts/run_supervised_search4.sh &   # background (recommended)

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search4_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search4_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════"
_log "  Supervised search4 run: ${SEARCH_ID}"
_log "  Dataset: 353k words (NLTK + competition, max_len=45)"
_log "  All experiments: curriculum(4×/2×), no early-game bias"
_log "  exp_j: 200k steps  (new-data baseline)"
_log "  exp_k: 250k steps"
_log "  exp_l: 300k steps"
_log "  prev best: exp_i 200k(4×/2×) → val=65.35%  test=68.12%  [251k dataset]"
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

run_exp "exp_j_200k_curr_4x2x" 200000
VAL_J="$LAST_VAL"; TEST_J="$LAST_TEST"

run_exp "exp_k_250k_curr_4x2x" 250000
VAL_K="$LAST_VAL"; TEST_K="$LAST_TEST"

run_exp "exp_l_300k_curr_4x2x" 300000
VAL_L="$LAST_VAL"; TEST_L="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  SEARCH4 COMPLETE — ${SEARCH_ID}"
_log "  Note: test set is larger (17,652 vs 12,574) — scores not directly comparable"
_log "═══════════════════════════════════════════════════════"
_log "  experiment                    val%       test%"
_log "  ──────────────────────────────────────────────────────"
_log "  [prev] exp_i 200k(4×/2×)      65.35%     68.12%  [251k dataset]"
_log "  exp_j  200k curr(4×/2×)        ${VAL_J}%     ${TEST_J}%"
_log "  exp_k  250k curr(4×/2×)        ${VAL_K}%     ${TEST_K}%"
_log "  exp_l  300k curr(4×/2×)        ${VAL_L}%     ${TEST_L}%"
_log "  ──────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "═══════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
