#!/usr/bin/env bash
# Supervised Search 7 — Dedicated Short-Word Model + Ensemble
#
# Context:
#   All training-side experiments (RL ×6, curriculum, scaling, data cleaning),
#   AND inference-time search have been exhausted. Best model: sup5_b test=68.79%.
#
#   Two remaining opportunities identified:
#     Short words (len 4–6): ceiling +6.54pp (if brought to average 68.79%)
#     Rare letters j/k/w:   ceiling +3.52pp  (already tried loss-reweight → +0.12pp)
#
#   The short-word problem is the bigger prize. 16×/4× curriculum FAILED because
#   it tried to train ONE model on all lengths — short/long conflict degraded both.
#
#   Solution: DEDICATED model trained ONLY on 4–6 letter words.
#   No trade-off. Full model capacity on short-word patterns.
#   At test time: route len≤6 to short model, len>6 to sup5_b (unchanged).
#
# Data:
#   data/train_short46.txt — 48,906 words of length 4–6 only.
#   (Created from train_clean1.txt by filtering for len ∈ {4,5,6}.)
#
# Experiments:
#   sw_a: d=256 L=4  3.25M params  150k steps  lr=3e-4  (fast, light)
#   sw_b: d=384 L=6  10.8M params  120k steps  lr=2e-4  (more capacity)
#
#   Both use the same 4×/2× curriculum *within* the short-word subset:
#     4-letter words weight=4.0, 5-letter words weight=2.0, 6-letter words weight=1.0
#   (Since all data is already "short", this just fine-tunes the balance within.)
#
# Evaluation:
#   Step 1 — Per-model short-word eval (len 4–6 test words, n=2,685)
#   Step 2 — Ensemble eval: sw_best + sup5_b, split at len=6
#             compare ensemble test score vs sup5_b alone (68.79%)
#
# Expected:
#   sw_a/b short-word rate: 35–55% (vs current 13.6/22.3/31.3% for len 4/5/6)
#   Ensemble overall:       70–72%  (+1.5–3pp over sup5_b)
#
# Usage:
#   nohup bash scripts/run_sup_search7.sh > /tmp/sup_search7_launcher.log 2>&1 &
#   tail -f /tmp/sup_search7_launcher.log

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search7_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search7_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"
SHORT_TRAIN="data/train_short46.txt"
VAL_FILE="data/val_words.txt"
TEST_FILE="data/test_words.txt"
SUP5B_CKPT="logs/sup_search5_20260521_0116/sup5_b_d512_L6/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# ── Verify data ───────────────────────────────────────────────────────────────
if [[ ! -f "$SHORT_TRAIN" ]]; then
    _log "Building train_short46.txt …"
    python3 -c "
words = open('data/train_clean1.txt').read().split()
short = [w for w in words if 4 <= len(w) <= 6]
open('data/train_short46.txt','w').write('\n'.join(short)+'\n')
print(f'train_short46.txt: {len(short):,} words')
"
fi
_log "train_short46.txt: $(wc -l < $SHORT_TRAIN) words"

# ── Header ────────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  Supervised Search 7 — Dedicated Short-Word Model: ${SEARCH_ID}"
_log "  Baseline: sup5_b  d=512 L=6  19.2M  test=68.79%"
_log "  Problem:  short words (4–6 letters) stuck at 13/22/31%"
_log "  Fix:      dedicated model trained ONLY on 4–6 letter words"
_log "  ─────────────────────────────────────────────────────────────────"
_log "  sw_a: d=256 L=4  3.25M   150k steps  lr=3e-4  (fast baseline)"
_log "  sw_b: d=384 L=6  10.8M   120k steps  lr=2e-4  (more capacity)"
_log "  ensemble: sw_best (len≤6) + sup5_b (len>6)"
_log "  target:   short words > 45%  ensemble overall > 70%"
_log "  est. runtime: ~3h + ~5h = ~8h total"
_log "═══════════════════════════════════════════════════════════════════"
_log ""

# ── run_experiment helper ──────────────────────────────────────────────────────
run_experiment() {
    local name="$1"; shift
    local ckpt_dir="${SEARCH_DIR}/${name}/checkpoints"
    local log_file="${SEARCH_DIR}/${name}/train.log"
    mkdir -p "$ckpt_dir"

    echo "started: $(date)" > "${SEARCH_DIR}/${name}/run_info.txt"
    _log "▶ Starting ${name}"

    python scripts/train_supervised.py \
        --train-file "$SHORT_TRAIN" \
        --val-file   "$VAL_FILE" \
        --ckpt-dir   "$ckpt_dir" \
        --log-file   "$log_file" \
        --length-curriculum \
        --short-weight  4.0 \
        --medium-weight 2.0 \
        --warmup  1000 \
        --bs      512 \
        --log-every  500 \
        --eval-every 5000 \
        --val-subset 5000 \
        --demo-every 10000 \
        --save-every 30000 \
        "$@"

    local best_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ ! -f "$best_ckpt" ]]; then
        _log "  ✗ No checkpoint saved — training did not improve val"
        return
    fi

    echo "finished: $(date)" >> "${SEARCH_DIR}/${name}/run_info.txt"
    _log "✓ Training done: ${name}"

    local best_val
    best_val=$(grep "best val win-rate:" "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+%' || echo "?")
    _log "  best val: ${best_val}"

    # ── Eval 1: short-word model alone on short test words ──────────────────
    _log "  Evaluating short model on short test words (len 4–6) …"
    python scripts/evaluate_ensemble.py \
        --ckpt-short "$best_ckpt" \
        --words      "$TEST_FILE" \
        --short-only \
        2>&1 | tee "${SEARCH_DIR}/${name}/short_eval.log" \
               | grep -E "WIN RATE|len=|weak|saved"

    local short_wr
    short_wr=$(grep "WIN RATE" "${SEARCH_DIR}/${name}/short_eval.log" \
        | grep -oP '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    _log "  ★ short-word win-rate (model alone): ${short_wr}"
    _log ""
}

# ── Experiments ───────────────────────────────────────────────────────────────

# sw_a: lightweight d=256 model — fast, serves as lower bound
run_experiment "sw_a_d256_L4" \
    --d-model  256 \
    --n-layers   4 \
    --n-heads    4 \
    --dim-ff  1024 \
    --lr      3e-4 \
    --steps  150000

# sw_b: larger d=384 model — more capacity for short-word patterns
run_experiment "sw_b_d384_L6" \
    --d-model  384 \
    --n-layers   6 \
    --n-heads    6 \
    --dim-ff  1536 \
    --lr      2e-4 \
    --steps  120000

# ── Ensemble evaluation ────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  ENSEMBLE EVALUATION — sw_best + sup5_b"
_log "═══════════════════════════════════════════════════════════════════"
_log ""

for sw_name in sw_a_d256_L4 sw_b_d384_L6; do
    local_ckpt="${SEARCH_DIR}/${sw_name}/checkpoints/supervised_best.pt"
    if [[ ! -f "$local_ckpt" ]]; then
        _log "  ✗ ${sw_name}: no checkpoint, skipping ensemble"
        continue
    fi

    for split in 6 7; do
        _log "  Ensemble: ${sw_name} + sup5_b  split=len${split} …"
        python scripts/evaluate_ensemble.py \
            --ckpt-short "$local_ckpt" \
            --ckpt-main  "$SUP5B_CKPT" \
            --words      "$TEST_FILE" \
            --split-len  "$split" \
            2>&1 | tee "${SEARCH_DIR}/${sw_name}/ensemble_split${split}.log" \
                   | grep -E "OVERALL|Short|Long|saved"

        local ens_wr
        ens_wr=$(grep "OVERALL WIN RATE" "${SEARCH_DIR}/${sw_name}/ensemble_split${split}.log" \
            | grep -oP '[0-9]+\.[0-9]+%' | head -1 || echo "?")
        _log "  ★ ensemble (split=${split}): ${ens_wr}  (baseline: 68.79%)"
        _log ""
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
_log "═══════════════════════════════════════════════════════════════════"
_log "  SUP SEARCH 7 COMPLETE — ${SEARCH_ID}"
_log "  baseline: sup5_b  d=512 L=6  19.2M  200k → test=68.79%"
_log "  Results in: ${SEARCH_DIR}/"
_log "═══════════════════════════════════════════════════════════════════"
