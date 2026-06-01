#!/usr/bin/env bash
# Supervised Search 8 — Architecture Scale-Up: d=768 and d=1024
#
# Context:
#   sup_search5 established the scaling law:
#     d=256  3.2M   → 66.20%  baseline
#     d=384  10.7M  → 68.24%  +1.92pp  (3.3× params)
#     d=512  18.9M  → 68.79%  +0.55pp  (1.8× params)  ← current best
#
#   All other approaches (RL ×6, curriculum, inference-time search,
#   dedicated short-word model) failed or gave <0.15pp.
#   Scaling is the only reliable lever remaining — diminishing but real.
#
#   Extrapolated expected gains (logarithmic scaling law):
#     d=768  42.5M  (~+0.2pp) → ~69.0%
#     d=1024 75.6M  (~+0.1pp) → ~69.1%
#   Neither will reach 70% alone, but together they confirm the ceiling
#   and may push slightly past 69%.
#
# Architecture:
#   d=768 : L=6  H=12  ff=3072  ~42.5M params
#   d=1024: L=6  H=16  ff=4096  ~75.6M params
#
# LR scaling (larger models need lower LR for stable training):
#   d=512  → lr=1e-4  (validated)
#   d=768  → lr=7e-5  (slight reduction)
#   d=1024 → lr=5e-5  (conservative for biggest model)
#
# Steps: 200k each (same as sup5_b — diminishing returns hit fast)
# Curriculum: 4×/2× (validated sweet spot from all prior searches)
# Data: train_clean1.txt (317k words, garbage removed)
#
# Estimated runtime:
#   d=768 : ~24h
#   d=1024: ~52h
#   Total : ~76h  (run sequentially in background)
#
# Usage:
#   nohup bash scripts/run_sup_search8.sh > /tmp/sup_search8_launcher.log 2>&1 &
#   tail -f /tmp/sup_search8_launcher.log

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search8_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search8_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"
CLEAN_TRAIN="data/train_clean1.txt"
VAL_FILE="data/val_words.txt"
TEST_FILE="data/test_words.txt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════════════════"
_log "  Supervised Search 8 — Architecture Scale-Up: ${SEARCH_ID}"
_log "  Baseline: sup5_b  d=512 L=6  18.9M  200k → test=68.79%"
_log "  Scaling law: +1.92pp (d=256→384), +0.55pp (d=384→512)"
_log "  Expected:    ~+0.2pp (d=512→768),  ~+0.1pp (d=768→1024)"
_log "  ─────────────────────────────────────────────────────────────────"
_log "  sup8_a: d=768  L=6 H=12 ff=3072  42.5M  200k steps  lr=7e-5"
_log "  sup8_b: d=1024 L=6 H=16 ff=4096  75.6M  200k steps  lr=5e-5"
_log "  est. runtime: ~24h + ~52h = ~76h total (sequential)"
_log "═══════════════════════════════════════════════════════════════════"
_log ""

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
        --short-weight  4.0 \
        --medium-weight 2.0 \
        --warmup  2000 \
        --bs      512 \
        --log-every  500 \
        --eval-every 5000 \
        --val-subset 5000 \
        --demo-every 10000 \
        --save-every 50000 \
        "$@"

    echo "finished: $(date)" >> "${SEARCH_DIR}/${name}/run_info.txt"

    local best_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ ! -f "$best_ckpt" ]]; then
        _log "  ✗ No checkpoint saved — training did not improve val"
        return
    fi

    _log "✓ Training done: ${name}"

    local best_val
    best_val=$(grep "best val win-rate:" "$log_file" | tail -1 \
        | grep -oP '[0-9]+\.[0-9]+%' || echo "?")
    _log "  best val: ${best_val}"

    _log "  Evaluating on test set …"
    python scripts/evaluate.py \
        --ckpt  "$best_ckpt" \
        --words "$TEST_FILE" \
        2>&1 | tee "${SEARCH_DIR}/${name}/test_eval.log" \
               | grep -E "TEST WIN RATE|len=.*weak|BIAS" | head -20

    local test_result
    test_result=$(grep "TEST WIN RATE" "${SEARCH_DIR}/${name}/test_eval.log" \
        | grep -oP '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    _log "  ★ test win-rate: ${test_result}  (baseline: 68.79%)"
    _log ""
}

# ── sup8_a: d=768 ─────────────────────────────────────────────────────────────
run_experiment "sup8_a_d768_L6" \
    --d-model  768 \
    --n-layers   6 \
    --n-heads   12 \
    --dim-ff  3072 \
    --lr      7e-5 \
    --steps  200000

# ── sup8_b: d=1024 ────────────────────────────────────────────────────────────
run_experiment "sup8_b_d1024_L6" \
    --d-model  1024 \
    --n-layers    6 \
    --n-heads    16 \
    --dim-ff   4096 \
    --lr       5e-5 \
    --steps   200000

# ── Summary ───────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  SUP SEARCH 8 COMPLETE — ${SEARCH_ID}"
_log "  baseline:  sup5_b  d=512   18.9M  test=68.79%"
_log "  sup8_a:    d=768   42.5M   test=?"
_log "  sup8_b:    d=1024  75.6M   test=?"
_log "  Results in: ${SEARCH_DIR}/"
_log "═══════════════════════════════════════════════════════════════════"
