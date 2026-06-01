#!/usr/bin/env bash
# RL search round 4 — GRPO (Group Relative Policy Optimization).
#
# Root cause of search3 failure (and all prior RL failures):
#   Batch-level advantage normalization (A = (RTG - batch_mean) / batch_std) conflates
#   word difficulty with action quality. Easy words always get high returns, hard words
#   (j/k/w/z) always get low returns — batch mean is word-difficulty noise, not signal.
#   Result: policy learns to avoid rare letters, making letter bias WORSE not better.
#
# GRPO fix:
#   Play K=4 games per word (128 unique words × 4 = 512 total, same batch size).
#   Advantage = RTG_k - mean(total_return_1..K for same word).
#   Word difficulty cancels out — gradient purely reflects which strategies beat
#   the average for THAT specific word.
#
# Experiments (sequential):
#   exp_rl4_a: GRPO K=4  lr=3e-6  aux=0.10  — core hypothesis test
#   exp_rl4_b: GRPO K=8  lr=3e-6  aux=0.10  — more games/word = lower variance
#   exp_rl4_c: GRPO K=4  lr=1e-5  aux=0.10  — higher LR once GRPO confirmed working
#
# Strategy:
#   Run (a) first. Check step-500 val against init 63.05%.
#   If (a) shows improvement → run (b) and (c).
#   If (a) val < 62% at step 500 → GRPO alone not enough → add value head (search5).
#
# Total wall time if all 3 run: ~3 × ~80min = ~4 hours
#
# Usage:
#   nohup bash scripts/run_rl_search4.sh &   # background, survive terminal close

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/rl_search4_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "rl_search4_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"
INIT_CKPT="logs/sup_search4_20260518_1149/exp_k_250k_curr_4x2x/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════════════"
_log "  RL search4 run: ${SEARCH_ID}"
_log "  Init: exp_k 250k → val=63.05%  test=66.20%"
_log "  Fix: GRPO — per-word relative advantage, removes word-difficulty confound"
_log "  Search3 failure: batch normalization = word difficulty noise, not signal"
_log "  ─────────────────────────────────────────────────────────────"
_log "  exp_rl4_a: GRPO K=4  lr=3e-6  aux=0.10  (core hypothesis)  6k steps"
_log "  exp_rl4_b: GRPO K=8  lr=3e-6  aux=0.10  (lower variance)   6k steps  [if (a) works]"
_log "  exp_rl4_c: GRPO K=4  lr=1e-5  aux=0.10  (higher LR)        6k steps  [if (a) works]"
_log "  target: val > 63.05% (supervised init) → rl_best.pt saved"
_log "═══════════════════════════════════════════════════════════════"

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
        --steps           6000            \
        --games-per-step  512             \
        --clip-eps        0.1             \
        --entropy-coef    0.005           \
        --ppo-epochs      1               \
        --ppo-minibatch   256             \
        --win-bonus       5.0             \
        --loss-penalty    -2.0            \
        --eval-every      100             \
        --demo-every      500             \
        --log-every       20              \
        --val-subset      2000            \
        --ckpt-dir        "$ckpt_dir"     \
        --log-file        "${exp_dir}/rl.log" \
        "${extra_flags[@]}"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    local val_pct
    val_pct=$(grep "saved rl_best.pt" "${exp_dir}/rl.log" \
              | tail -1 | grep -oP 'val=\K[0-9.]+' || echo "n/a")
    if [[ "$val_pct" == "n/a" ]]; then
        val_pct=$(grep "val_win_rate=" "${exp_dir}/rl.log" \
                  | tail -1 | grep -oP 'val_win_rate=\K[0-9.]+' || echo "n/a")
        _log "  best val: ${val_pct}%  (RL did not improve over supervised init)"
    else
        _log "  best val: ${val_pct}%  ← rl_best.pt saved ✓"
    fi
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"

    local eval_ckpt="${ckpt_dir}/rl_best.pt"
    if [[ ! -f "$eval_ckpt" ]]; then
        eval_ckpt="$INIT_CKPT"
        _log "  No rl_best.pt — evaluating supervised init for comparison"
    fi

    _log "  Running test evaluation…"
    python scripts/evaluate.py \
        --ckpt     "$eval_ckpt"        \
        --words    data/test_words.txt \
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

# ── Check step-500 val and decide whether to continue ────────────────────────
check_and_gate() {
    local log_file="$1"
    local step500_val
    step500_val=$(grep "eval ppo step=500" "$log_file" \
                  | grep -oP 'val_win_rate=\K[0-9.]+' | tail -1 || echo "")
    if [[ -z "$step500_val" ]]; then
        _log "  ⚠ step-500 eval not found — running remaining experiments anyway"
        GATE_PASS=1; return
    fi
    if (( $(echo "$step500_val < 62.0" | bc -l) )); then
        _log "  ✗ GATE FAIL: step-500 val=${step500_val}% < 62%"
        _log "  ✗ GRPO alone not enough. Skipping (b) and (c)."
        _log "  ✗ Next step: add value head (critic) to HangmanPolicy → search5"
        GATE_PASS=0
    else
        _log "  ✓ GATE PASS: step-500 val=${step500_val}% ≥ 62% — running (b) and (c)"
        GATE_PASS=1
    fi
}

# ── Run experiments ───────────────────────────────────────────────────────────

GATE_PASS=1

run_exp "exp_rl4_a_grpo_k4"  --lr 3e-6  --aux-coef 0.10  --grpo-k 4
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"
check_and_gate "${SEARCH_DIR}/exp_rl4_a_grpo_k4/rl.log"

VAL_B="skipped"; TEST_B="skipped"
VAL_C="skipped"; TEST_C="skipped"

if [[ "$GATE_PASS" -eq 1 ]]; then
    run_exp "exp_rl4_b_grpo_k8"  --lr 3e-6  --aux-coef 0.10  --grpo-k 8
    VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"

    run_exp "exp_rl4_c_grpo_k4_lr1e5"  --lr 1e-5  --aux-coef 0.10  --grpo-k 4
    VAL_C="$LAST_VAL"; TEST_C="$LAST_TEST"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════════════"
_log "  RL SEARCH4 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════════════"
_log "  supervised init (exp_k)              val=63.05%  test=66.20%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  experiment                           val%        test%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  exp_rl4_a GRPO K=4  lr=3e-6         ${VAL_A}%      ${TEST_A}%"
_log "  exp_rl4_b GRPO K=8  lr=3e-6         ${VAL_B}        ${TEST_B}"
_log "  exp_rl4_c GRPO K=4  lr=1e-5         ${VAL_C}        ${TEST_C}"
_log "  ─────────────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "  next (if all fail): add value head → scripts/run_rl_search5.sh"
_log "═══════════════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
