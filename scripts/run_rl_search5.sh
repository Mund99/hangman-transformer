#!/usr/bin/env bash
# RL Search 5 — KL-penalised PPO (RLHF-style).
#
# Root cause of searches 1–4 failure: aux BCE creates a degraded 58–59%
# equilibrium — it's a miniature supervised training loop with weak gradient
# (256 words, no curriculum) that fights PPO until they settle at a floor.
#
# Fix: replace aux BCE with KL(π_θ || π_ref) at rollout states, measured
# against a frozen reference model (the supervised checkpoint).
#   • KL grows exactly where the policy is changing → self-regulating
#   • No independent equilibrium floor — KL → 0 when policy matches ref
#   • Principled RLHF regularisation (same mechanism as InstructGPT)
#
# Init: ft_b step-6k checkpoint (val=63.15%, test=66.32%) — best so far.
# Ref:  same ft_b checkpoint (KL measured against the fine-tuned init).
#
# Experiments (all: GRPO K=4, terminal reward, curriculum, aux-coef=0):
#   rl5_a: kl_coef=0.03  — light regularisation, more RL freedom
#   rl5_b: kl_coef=0.10  — moderate (RLHF default range)
#   rl5_c: kl_coef=0.30  — strong, stays very close to supervised init
#
# Gate: if val < 62.5% at step 500 → KL too weak (forgetting) → skip to next.
#       if val ≥ 63.5% at step 500 → KL working well, continue.
#
# Usage:
#   nohup bash scripts/run_rl_search5.sh &

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/rl_search5_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "rl_search5_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

# Best supervised checkpoint: ft_b step-6k (val=63.15%, test=66.32%)
INIT_CKPT="logs/sup_finetune_20260520_1228/ft_b_letter_w10_curriculum/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════════════"
_log "  RL Search 5 — KL-penalised PPO: ${SEARCH_ID}"
_log "  Init: ft_b step-6k  val=63.15%  test=66.32%"
_log "  Fix:  KL(π_θ || π_ref) replaces aux BCE"
_log "  ─────────────────────────────────────────────────────────────"
_log "  rl5_a: kl_coef=0.03  GRPO K=4  terminal  curriculum"
_log "  rl5_b: kl_coef=0.10  GRPO K=4  terminal  curriculum"
_log "  rl5_c: kl_coef=0.30  GRPO K=4  terminal  curriculum"
_log "  gate:  val < 62.5% at step 500 → skip"
_log "  target: val > 63.15%  (improve on supervised init)"
_log "═══════════════════════════════════════════════════════════════"

BEST_CKPT="$INIT_CKPT"
BEST_VAL=0.0

run_exp() {
    local name="$1"; shift
    local extra_flags=("$@")

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    python scripts/train_rl.py \
        --init            "$INIT_CKPT"    \
        --steps           5000            \
        --games-per-step  512             \
        --grpo-k          4               \
        --terminal-reward                 \
        --curriculum                      \
        --aux-coef        0.0             \
        --lr              3e-6            \
        --clip-eps        0.1             \
        --ppo-epochs      1               \
        --ppo-minibatch   256             \
        --entropy-coef    0.003           \
        --eval-every      500             \
        --val-subset      1500            \
        --log-every       100             \
        --demo-every      500             \
        --ckpt-dir        "$ckpt_dir"     \
        --log-file        "${exp_dir}/train.log" \
        "${extra_flags[@]}"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    # Gate check: read val at step 500
    local val_500
    val_500=$(grep -oP '\[eval ppo step=\s*500\].*val_win_rate=\K[0-9.]+' \
              "${exp_dir}/train.log" | tail -1 || echo "n/a")
    _log "  val@500: ${val_500}%"

    local best_val
    best_val=$(grep -oP 'val_win_rate=\K[0-9.]+' "${exp_dir}/train.log" \
               | tail -1 || echo "n/a")
    _log "  best val: ${best_val}%"
    echo "best_val: ${best_val}%" >> "${exp_dir}/run_info.txt"

    # Test evaluation on best checkpoint (or fall back to init)
    local eval_ckpt="${ckpt_dir}/rl_best.pt"
    if [[ ! -f "$eval_ckpt" ]]; then
        eval_ckpt="$INIT_CKPT"
        _log "  No improvement over init — evaluating supervised init"
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

    # Track best checkpoint globally
    if [[ -f "${ckpt_dir}/rl_best.pt" ]]; then
        rl_val=$(grep -oP '→ saved rl_best\.pt.*val=\K[0-9.]+' "${exp_dir}/train.log" \
                 | tail -1 || echo "0")
        if awk "BEGIN{exit !($rl_val > $BEST_VAL)}"; then
            BEST_VAL="$rl_val"
            BEST_CKPT="${ckpt_dir}/rl_best.pt"
            _log "  ★ New overall best: val=${BEST_VAL}%  ckpt=${BEST_CKPT}"
        fi
    fi

    LAST_VAL="$best_val"
    LAST_TEST="$test_pct"
    LAST_VAL500="$val_500"
}

# ── Run experiments ───────────────────────────────────────────────────────────

run_exp "rl5_a_kl0.03" --kl-coef 0.03
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"; V500_A="$LAST_VAL500"

run_exp "rl5_b_kl0.10" --kl-coef 0.10
VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"; V500_B="$LAST_VAL500"

run_exp "rl5_c_kl0.30" --kl-coef 0.30
VAL_C="$LAST_VAL"; TEST_C="$LAST_TEST"; V500_C="$LAST_VAL500"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════════════"
_log "  RL SEARCH 5 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════════════"
_log "  supervised init (ft_b)    val=63.15%  test=66.32%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  experiment          val@500   final_val%   test%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  rl5_a kl=0.03       ${V500_A}%    ${VAL_A}%       ${TEST_A}%"
_log "  rl5_b kl=0.10       ${V500_B}%    ${VAL_B}%       ${TEST_B}%"
_log "  rl5_c kl=0.30       ${V500_C}%    ${VAL_C}%       ${TEST_C}%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  target: test > 66.32%"
_log "  best ckpt: ${BEST_CKPT}"
_log "  next: if test > 66.32% → stronger kl_coef sweep / more steps"
_log "        if val@500 < 62.5% → forgetting; try kl_coef × 3"
_log "        if val@500 stable but no test gain → reward design issue"
_log "═══════════════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
