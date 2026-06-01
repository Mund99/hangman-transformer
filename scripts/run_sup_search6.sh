#!/usr/bin/env bash
# Supervised Search 6 — Short-Word Focus
#
# Context:
#   sup_search5 proved architecture scaling works but hits diminishing returns:
#     sup5_a (d=384, 10.8M) → test=68.24%  (+2.19pp)
#     sup5_b (d=512, 19.2M) → test=68.79%  (+0.55pp)
#   Short words (4–6 letters) stuck at ~13/22/31% across ALL models despite 4×/2× curriculum.
#   Fixing short words alone = +5.2pp potential → could reach ~74%.
#
# Hypothesis:
#   The 4×/2× curriculum is not aggressive enough — short words are still
#   overwhelmed by the volume of medium/long words. Raising to 16×/4× should
#   force the model to deeply specialise on short-word letter patterns.
#
# Experiments:
#   sup6_a: fine-tune from sup5_b best (d=512, test=68.79%) with 16×/4× curriculum
#           Low LR=5e-5 to preserve long-word knowledge while improving short words.
#           100k steps — targeted fine-tune, not full retraining.
#
#   sup6_b: fresh d=512 training with 16×/4× curriculum from scratch
#           Baseline comparison: does aggressive curriculum help even without warm start?
#           250k steps, same architecture as sup5_b.
#
# Expected outcomes:
#   sup6_a: if short-word curriculum fine-tune works → +2–4pp test
#   sup6_b: if fresh training with aggressive curriculum > sup5_b → confirms curriculum is key
#   If both ~= 68.79%: short-word problem is not a curriculum issue (need different approach)
#
# Usage:
#   nohup bash scripts/run_sup_search6.sh > /tmp/sup_search6_launcher.log 2>&1 &
#   tail -f /tmp/sup_search6_launcher.log

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search6_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search6_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"
CLEAN_TRAIN="data/train_clean1.txt"
VAL_FILE="data/val_words.txt"
TEST_FILE="data/test_words.txt"
SUP5B_CKPT="logs/sup_search5_20260521_0116/sup5_b_d512_L6/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# ── Header ────────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  Supervised Search 6 — Short-Word Focus: ${SEARCH_ID}"
_log "  Baseline: sup5_b  d=512 L=6  19.2M params  200k steps  test=68.79%"
_log "  Problem:  short words (4–6 letters) stuck at 13/22/31% across all models"
_log "  Fix:      raise curriculum from 4×/2× to 16×/4×"
_log "  ─────────────────────────────────────────────────────────────────"
_log "  sup6_a: FINE-TUNE sup5_b  16×/4× curriculum  lr=5e-5   100k steps"
_log "  sup6_b: FRESH d=512       16×/4× curriculum  lr=1e-4   250k steps"
_log "  target: short words > 40%  overall test > 70%"
_log "  est. runtime: ~4 hrs + ~11 hrs = ~15 hrs total"
_log "═══════════════════════════════════════════════════════════════════"
_log ""

# ── run_experiment helper ─────────────────────────────────────────────────────
run_experiment() {
    local name="$1"; shift
    local ckpt_dir="${SEARCH_DIR}/${name}/checkpoints"
    local log_file="${SEARCH_DIR}/${name}/train.log"
    mkdir -p "$ckpt_dir"

    echo "started: $(date)" > "${SEARCH_DIR}/${name}/run_info.txt"

    _log "▶ Starting ${name}"
    python scripts/train_supervised.py \
        --train-file "$CLEAN_TRAIN" \
        --val-file   "$VAL_FILE" \
        --ckpt-dir   "$ckpt_dir" \
        --log-file   "$log_file" \
        --length-curriculum \
        --warmup  2000 \
        --bs      512 \
        --log-every  500 \
        --eval-every 5000 \
        --val-subset 5000 \
        --demo-every 10000 \
        --save-every 50000 \
        "$@"

    local best_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ ! -f "$best_ckpt" ]]; then
        _log "  ✗ No checkpoint saved — training did not improve val"
        return
    fi

    _log "✓ Training done: ${name}"

    local best_val
    best_val=$(grep "best val win-rate:" "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+%' || echo "?")
    _log "  best val: ${best_val}"

    _log "  Evaluating best checkpoint on test set…"
    python scripts/evaluate.py \
        --ckpt  "$best_ckpt" \
        --words "$TEST_FILE" \
        2>&1 | tee "${SEARCH_DIR}/${name}/test_eval.log" \
             | grep -E "TEST WIN RATE|◄ weak|len=|BIAS" | head -30
    local test_result
    test_result=$(grep "TEST WIN RATE" "${SEARCH_DIR}/${name}/test_eval.log" \
        | grep -oP '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    _log "  ★ test win-rate: ${test_result}"
    _log ""
}

# ── Experiments ───────────────────────────────────────────────────────────────

# sup6_a: fine-tune from sup5_b with aggressive short-word curriculum
run_experiment "sup6_a_finetune_16x4x" \
    --init        "$SUP5B_CKPT" \
    --short-weight  16.0 \
    --medium-weight  4.0 \
    --lr          5e-5 \
    --steps      100000

# sup6_b: fresh d=512 with aggressive short-word curriculum
run_experiment "sup6_b_fresh_16x4x" \
    --d-model  512 \
    --n-layers   6 \
    --n-heads    8 \
    --dim-ff  2048 \
    --short-weight  16.0 \
    --medium-weight  4.0 \
    --lr      1e-4 \
    --steps  250000

# ── Summary ───────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  SUP SEARCH 6 COMPLETE — ${SEARCH_ID}"
_log "  baseline: sup5_b  d=512 L=6  19.2M  200k → test=68.79%"
_log "  Results in: ${SEARCH_DIR}/"
_log "═══════════════════════════════════════════════════════════════════"
