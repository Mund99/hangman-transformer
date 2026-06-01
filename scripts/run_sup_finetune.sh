#!/usr/bin/env bash
# Targeted supervised fine-tuning from exp_k (val=63.05%, test=66.20%).
#
# Hypothesis: the j/k/w/z letter bias is a supervised problem, not an RL problem.
# The model under-weights rare letters because they appear in only ~2-3% of words —
# sparse BCE signal during pretraining. Fix: oversample those words heavily.
#
# Experiments:
#   ft_a: rare-letter-weight=5  + length-curriculum (4×/2×)  lr=1e-5  30k steps
#         Upweights j/k/w/z IN THE LOSS (not the sampler) — preserves word distribution.
#         Curriculum keeps short-word gradient healthy alongside rare-letter fix.
#   ft_b: rare-letter-weight=10 + length-curriculum (4×/2×)  lr=1e-5  30k steps
#         Stronger rare-letter signal — test whether 10× is stable at lower LR.
#
# Key lesson from v1 (killed): --rare-weight=10 (word sampler) caused catastrophic
# forgetting — 63.05% → 55.90% in 8k steps. Word oversampling biases ~71% of
# gradient toward rare-letter words, destroying common-letter knowledge.
# Fix: per-letter loss weighting keeps word distribution natural, only amplifies
# the rare-letter signal within each batch.
#
# Fine-tuning config:
#   lr: 1e-5 (vs 3e-4 original) — 30× lower than pretraining, fresh restart after cosine decay
#   steps: 30k, warmup: 500, bs: 512 (same)
#
# Expected: +3-5pp on j/k/w/z win rates, minimal regression on common letters.
#
# Usage:
#   nohup bash scripts/run_sup_finetune.sh &   # background, survive terminal close

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_finetune_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_finetune_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"
INIT_CKPT="logs/sup_search4_20260518_1149/exp_k_250k_curr_4x2x/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════════════"
_log "  Supervised fine-tune run: ${SEARCH_ID}"
_log "  Init: exp_k 250k → val=63.05%  test=66.20%"
_log "  Goal: fix j/k/w/z letter bias via targeted oversampling"
_log "  ─────────────────────────────────────────────────────────────"
_log "  ft_a: rare-letter-weight=5  + curriculum  lr=1e-5  30k steps"
_log "  ft_b: rare-letter-weight=10 + curriculum  lr=1e-5  30k steps"
_log "  target: test > 66.20%  (improve on supervised init)"
_log "═══════════════════════════════════════════════════════════════"

run_ft() {
    local name="$1"; shift
    local extra_flags=("$@")

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    python scripts/train_supervised.py \
        --init        "$INIT_CKPT"    \
        --steps       30000           \
        --warmup      500             \
        --bs          512             \
        --lr          1e-5            \
        --eval-every  2000            \
        --val-subset  2000            \
        --demo-every  5000            \
        --log-every   500             \
        --ckpt-dir    "$ckpt_dir"     \
        --log-file    "${exp_dir}/train.log" \
        "${extra_flags[@]}"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    local val_pct
    val_pct=$(grep -oP 'val_win_rate=\K[0-9.]+' "${exp_dir}/train.log" \
              | tail -1 || echo "n/a")
    _log "  best val: ${val_pct}%"
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"

    _log "  Running test evaluation…"
    local eval_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ ! -f "$eval_ckpt" ]]; then
        eval_ckpt="$INIT_CKPT"
        _log "  No improvement over init — evaluating supervised init for comparison"
    fi
    python scripts/evaluate.py \
        --ckpt     "$eval_ckpt"          \
        --words    data/test_words.txt   \
        --log-file "${exp_dir}/eval_test.log"

    local test_pct
    test_pct=$(grep -oP 'TEST WIN RATE\s+:\s+\K[0-9.]+' "${exp_dir}/eval_test.log" \
               | tail -1 || echo "n/a")
    echo "test_win_rate: ${test_pct}%" >> "${exp_dir}/run_info.txt"
    echo "eval_done: $(date)"         >> "${exp_dir}/run_info.txt"
    _log "  test win-rate: ${test_pct}%"

    LAST_VAL="$val_pct"
    LAST_TEST="$test_pct"
}

# ── Run experiments ───────────────────────────────────────────────────────────

run_ft "ft_a_letter_w5_curriculum"  \
    --rare-letter-weight 5.0  --length-curriculum --short-weight 4.0 --medium-weight 2.0
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"

run_ft "ft_b_letter_w10_curriculum" \
    --rare-letter-weight 10.0 --length-curriculum --short-weight 4.0 --medium-weight 2.0
VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════════════"
_log "  SUP FINETUNE COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════════════"
_log "  supervised init (exp_k)             val=63.05%  test=66.20%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  experiment                          val%        test%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  ft_a letter-w=5  + curriculum        ${VAL_A}%      ${TEST_A}%"
_log "  ft_b letter-w=10 + curriculum       ${VAL_B}%      ${TEST_B}%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  target: test > 66.20%"
_log "  next: if improved → try RL (KL-PPO) on top of best ft ckpt"
_log "        if regressed → rare_weight too high, try 5.0"
_log "═══════════════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
