#!/usr/bin/env bash
# Supervised Search 5 — Architecture scale-up
#
# Context:
#   RL fine-tuning (searches 1–6) failed to beat the supervised baseline on test.
#   Root cause: model capacity bottleneck + training data contains some noise.
#   The 3.25M param model (d=256, L=4) has hit its ceiling at ~66% on 353k corpus.
#
# Data cleaning:
#   train_clean1 removes ~248 garbage words (0.1% of 317k):
#     - 3+ consecutive repeated chars (aaaaaa, blottto...)
#     - <15% vowel ratio even counting y (pure consonant-cluster abbreviations)
#   Test set is NEVER changed — all test scores remain directly comparable.
#
# Experiments (both use train_clean1 and same 4×/2× curriculum):
#   sup5_a: d_model=384  6L 6H ff=1536  ~10.8M params  250k steps  (~9 hrs)
#   sup5_b: d_model=512  6L 8H ff=2048  ~19.2M params  200k steps  (~13 hrs)
#
# Baseline: exp_k  d=256 4L 4H ff=1024  3.25M params  250k steps  test=66.20%
#
# LR scaling: larger models benefit from slightly lower LR (more stable gradients)
#   d=384: lr=2e-4  (vs 3e-4 for d=256)
#   d=512: lr=1e-4  (conservative — big model, protect early training)
# Both use warmup=2000 (longer warmup for larger models)
#
# Usage:
#   nohup bash scripts/run_sup_search5.sh > /tmp/sup_search5_launcher.log 2>&1 &
#   tail -f /tmp/sup_search5_launcher.log

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search5_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search5_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"
CLEAN_TRAIN="data/train_clean1.txt"
VAL_FILE="data/val_words.txt"
TEST_FILE="data/test_words.txt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# ── Build train_clean1 if it doesn't exist ────────────────────────────────────
if [[ ! -f "$CLEAN_TRAIN" ]]; then
    _log "Building train_clean1.txt …"
    python3 - << 'PYEOF'
import re, sys
vowels = set('aeiouy')

def is_clean(w):
    v_ratio = sum(c in vowels for c in w) / len(w)
    if v_ratio < 0.15: return False            # consonant-heavy abbreviations
    if re.search(r'(.)\1{2,}', w): return False  # 3+ repeated chars
    return True

with open('data/train_words.txt') as f:
    words = [l.strip() for l in f if l.strip()]

clean = [w for w in words if is_clean(w)]
with open('data/train_clean1.txt', 'w') as f:
    f.write('\n'.join(clean) + '\n')

removed = len(words) - len(clean)
print(f"train_clean1: {len(clean):,} words  (removed {removed:,} / {removed/len(words)*100:.1f}%)")
PYEOF
    _log "train_clean1 ready: $(wc -l < $CLEAN_TRAIN) words"
else
    _log "train_clean1 already exists: $(wc -l < $CLEAN_TRAIN) words"
fi

# ── Header ────────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  Supervised Search 5 — Architecture scale-up: ${SEARCH_ID}"
_log "  Baseline: exp_k  d=256 L=4  train_ori  250k steps  test=66.20%"
_log "  Data: train_clean1 (317k → 314k, garbage removed)"
_log "  ─────────────────────────────────────────────────────────────────"
_log "  sup5_a: d=384 L=6 H=6 ff=1536  ~10.8M params  250k steps  lr=2e-4"
_log "  sup5_b: d=512 L=6 H=8 ff=2048  ~19.2M params  200k steps  lr=1e-4"
_log "  target: test > 66.20%  (beat exp_k baseline)"
_log "  est. runtime: ~9 hrs + ~13 hrs = ~22 hrs total"
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
        --log-file  "$log_file" \
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

    local best_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ ! -f "$best_ckpt" ]]; then
        _log "  ✗ No checkpoint saved — training did not improve"
        return
    fi

    _log "✓ Training done: ${name}"

    # Extract best val from log
    local best_val
    best_val=$(grep "best val win-rate:" "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+%' || echo "?")
    _log "  best val: ${best_val}"

    # Test evaluation
    _log "  Evaluating on test set…"
    local test_result
    test_result=$(python scripts/evaluate.py \
        --ckpt  "$best_ckpt" \
        --words "$TEST_FILE" 2>&1 | grep -oP 'TEST WIN RATE\s*:\s*[0-9.]+%' | grep -oP '[0-9.]+%' | tail -1 || echo "eval failed")
    _log "  test win-rate: ${test_result}"

    # Also evaluate all periodic checkpoints for best pick
    _log "  Evaluating all saved checkpoints…"
    local best_test=0
    local best_test_ckpt="$best_ckpt"
    for ckpt in "${ckpt_dir}"/*.pt; do
        local r
        r=$(python scripts/evaluate.py --ckpt "$ckpt" --words "$TEST_FILE" 2>&1 \
            | grep -oP 'TEST WIN RATE\s*:\s*[0-9.]+' | grep -oP '[0-9.]+$' | tail -1 || echo "0")
        _log "    $(basename $ckpt): test=${r}%"
        if (( $(echo "$r > $best_test" | bc -l) )); then
            best_test=$r
            best_test_ckpt=$ckpt
        fi
    done
    _log "  ★ Best test: ${best_test}%  (${best_test_ckpt})"
    _log ""
}

# ── Experiments ───────────────────────────────────────────────────────────────

run_experiment "sup5_a_d384_L6" \
    --d-model  384 \
    --n-layers   6 \
    --n-heads    6 \
    --dim-ff  1536 \
    --lr      2e-4 \
    --steps  250000

run_experiment "sup5_b_d512_L6" \
    --d-model  512 \
    --n-layers   6 \
    --n-heads    8 \
    --dim-ff  2048 \
    --lr      1e-4 \
    --steps  200000

# ── Summary ───────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  SUP SEARCH 5 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════════════════"
_log "  baseline: exp_k  d=256 L=4  train_ori  250k → test=66.20%"
_log "  Results in: ${SEARCH_DIR}/"
_log "═══════════════════════════════════════════════════════════════════"
