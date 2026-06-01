#!/usr/bin/env bash
# RL search round 2 — PPO fine-tuning from exp_k (66.20%), fixing search1 root causes.
#
# Init checkpoint: exp_k 250k steps, 353k dataset  → val=63.05%  test=66.20%
#
# Search1 root causes (all fixed here):
#   1. Curriculum mismatch    → NO --curriculum (natural word distribution)
#   2. Too many PPO epochs    → --ppo-epochs 1  (fresh importance ratios)
#   3. Small batch / noisy G  → --games-per-step 512 (2× vs search1, ppo-minibatch=256)
#   4. LR too high            → lr=3e-6 (conservative, protect supervised knowledge)
#   5. Shaped reward conflict → --terminal-reward (+1/-1 win/loss only)
#
# Experiments:
#   exp_rl2_a: baseline with all 5 fixes  (lr=3e-6  aux=0.20  6k steps)
#   exp_rl2_b: weaker supervised anchor   (lr=3e-6  aux=0.05  6k steps)  more RL freedom
#   exp_rl2_c: stronger supervised anchor (lr=3e-6  aux=0.40  6k steps)  safer, slower
#
# Usage:
#   bash scripts/run_rl_search2.sh
#   bash scripts/run_rl_search2.sh &   # background (recommended)
#
# Abort criteria (check at step 500 of exp_rl2_a):
#   FAIL if val_win_rate < 62% or win_ma flatlines at same level as step 20
#   SUCCESS if val_win_rate >= 63.05% (supervised init) at any point

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/rl_search2_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "rl_search2_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

INIT_CKPT="logs/sup_search4_20260518_1149/exp_k_250k_curr_4x2x/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════"
_log "  RL search2 run: ${SEARCH_ID}"
_log "  Init: exp_k 250k (search4) → test=66.20%  val=63.05%"
_log "  All fixes applied vs search1:"
_log "    1. No curriculum  (natural word distribution)"
_log "    2. ppo-epochs=1   (fresh importance ratios)"
_log "    3. games=1024     (4× larger batch, lower variance)"
_log "    4. lr=3e-6        (conservative, protect supervised)"
_log "    5. terminal-reward (+1/-1 win/loss only)"
_log "  exp_rl2_a: aux=0.20  6k steps  (all-fixes baseline)"
_log "  exp_rl2_b: aux=0.05  6k steps  (weaker anchor, more RL)"
_log "  exp_rl2_c: aux=0.40  6k steps  (stronger anchor, safer)"
_log "  target: ≥70% test win-rate"
_log "  abort: if exp_rl2_a val < 62% at step 500"
_log "═══════════════════════════════════════════════════════"

run_exp() {
    local name="$1"
    local aux="$2"
    local steps="$3"
    local lr="${4:-3e-6}"

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}  (lr=${lr}  aux=${aux}  steps=${steps})"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    python scripts/train_rl.py \
        --init            "$INIT_CKPT" \
        --steps           "$steps"     \
        --games-per-step  512          \
        --lr              "$lr"        \
        --clip-eps        0.1          \
        --aux-coef        "$aux"       \
        --entropy-coef    0.005        \
        --ppo-epochs      1            \
        --ppo-minibatch   256          \
        --terminal-reward              \
        --eval-every      100          \
        --demo-every      500          \
        --log-every       20           \
        --val-subset      2000         \
        --ckpt-dir        "$ckpt_dir"  \
        --log-file        "${exp_dir}/rl.log"
        # NO --curriculum (natural word distribution)

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    # best val from rl.log
    local val_pct
    val_pct=$(grep "saved rl_best.pt" "${exp_dir}/rl.log" \
              | tail -1 | grep -oP 'val=\K[0-9.]+' || echo "n/a")
    if [[ "$val_pct" == "n/a" ]]; then
        val_pct=$(grep "val_win_rate=" "${exp_dir}/rl.log" \
                  | tail -1 | grep -oP 'val_win_rate=\K[0-9.]+' || echo "n/a")
        _log "  best val: ${val_pct}%  (RL did not improve over supervised init)"
    else
        _log "  best val: ${val_pct}%  ← rl_best.pt saved!"
    fi
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"

    # evaluate best checkpoint (rl_best.pt if saved, else fall back to init)
    local eval_ckpt="${ckpt_dir}/rl_best.pt"
    if [[ ! -f "$eval_ckpt" ]]; then
        eval_ckpt="$INIT_CKPT"
        _log "  No rl_best.pt — evaluating supervised init for comparison"
    fi

    _log "  Running test evaluation…"
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

# ── Run experiments sequentially ─────────────────────────────────────────────

run_exp "exp_rl2_a_baseline"       0.20  6000
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"

run_exp "exp_rl2_b_weak_anchor"    0.05  6000
VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"

run_exp "exp_rl2_c_strong_anchor"  0.40  6000
VAL_C="$LAST_VAL"; TEST_C="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  RL SEARCH2 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════"
_log "  init (exp_k supervised)              val=63.05%  test=66.20%"
_log "  ─────────────────────────────────────────────────────"
_log "  experiment                      val%        test%"
_log "  ─────────────────────────────────────────────────────"
_log "  exp_rl2_a all-fixes baseline     ${VAL_A}%      ${TEST_A}%"
_log "  exp_rl2_b weak anchor (aux=0.05) ${VAL_B}%      ${TEST_B}%"
_log "  exp_rl2_c strong anchor (aux=0.4)${VAL_C}%      ${TEST_C}%"
_log "  ─────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "═══════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
