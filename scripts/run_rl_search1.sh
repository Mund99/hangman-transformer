#!/usr/bin/env bash
# RL search round 1 — PPO fine-tuning from search4 exp_k checkpoint (66.20%).
#
# Init checkpoint: exp_k 250k steps, 353k dataset
# All experiments: curriculum (35% short / 35% medium / 30% long), 256 games/step
#
# What we vary:
#   exp_rl_a: baseline          lr=1e-5  clip=0.2  aux=0.10  8k steps
#   exp_rl_b: conservative      lr=5e-6  clip=0.1  aux=0.30  8k steps  (preserve sup. knowledge)
#   exp_rl_c: stronger push     lr=2e-5  clip=0.2  aux=0.20  10k steps (more steps + balanced aux)
#
# Usage:
#   bash scripts/run_rl_search1.sh
#   bash scripts/run_rl_search1.sh &   # background (recommended)

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/rl_search1_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "rl_search1_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

INIT_CKPT="logs/sup_search4_20260518_1149/exp_k_250k_curr_4x2x/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════"
_log "  RL search1 run: ${SEARCH_ID}"
_log "  Init: exp_k 250k (search4) → test=66.20%  val=63.05%"
_log "  exp_rl_a: lr=1e-5  clip=0.2  aux=0.10  8k steps  (baseline)"
_log "  exp_rl_b: lr=5e-6  clip=0.1  aux=0.30  8k steps  (conservative)"
_log "  exp_rl_c: lr=2e-5  clip=0.2  aux=0.20  10k steps (stronger push)"
_log "  target: ≥70% test win-rate"
_log "═══════════════════════════════════════════════════════"

run_exp() {
    local name="$1"
    local lr="$2"
    local clip="$3"
    local aux="$4"
    local steps="$5"

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}  (lr=${lr}  clip=${clip}  aux=${aux}  steps=${steps})"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    python scripts/train_rl.py \
        --init           "$INIT_CKPT" \
        --steps          "$steps"     \
        --games-per-step 256          \
        --lr             "$lr"        \
        --clip-eps       "$clip"      \
        --aux-coef       "$aux"       \
        --entropy-coef   0.003        \
        --ppo-epochs     4            \
        --curriculum                  \
        --eval-every     100          \
        --demo-every     200          \
        --log-every      20           \
        --val-subset     2000         \
        --ckpt-dir       "$ckpt_dir"  \
        --log-file       "${exp_dir}/rl.log"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    # best val from rl.log
    local val_pct
    val_pct=$(grep "saved rl_best.pt" "${exp_dir}/rl.log" \
              | tail -1 | grep -oP 'val=\K[0-9.]+' || echo "n/a")
    if [[ "$val_pct" == "n/a" ]]; then
        # RL never improved over supervised init — report final eval line instead
        val_pct=$(grep "val_win_rate=" "${exp_dir}/rl.log" \
                  | tail -1 | grep -oP 'val_win_rate=\K[0-9.]+' || echo "n/a")
        _log "  best val: ${val_pct}%  (RL did not improve over supervised init)"
    else
        _log "  best val: ${val_pct}%"
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

run_exp "exp_rl_a_baseline"     1e-5  0.2  0.10  8000
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"

run_exp "exp_rl_b_conservative" 5e-6  0.1  0.30  8000
VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"

run_exp "exp_rl_c_push"         2e-5  0.2  0.20  10000
VAL_C="$LAST_VAL"; TEST_C="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  RL SEARCH1 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════"
_log "  init (exp_k supervised)              val=63.05%  test=66.20%"
_log "  ─────────────────────────────────────────────────────"
_log "  experiment                  val%        test%"
_log "  ─────────────────────────────────────────────────────"
_log "  exp_rl_a baseline            ${VAL_A}%      ${TEST_A}%"
_log "  exp_rl_b conservative        ${VAL_B}%      ${TEST_B}%"
_log "  exp_rl_c push                ${VAL_C}%      ${TEST_C}%"
_log "  ─────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "═══════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
