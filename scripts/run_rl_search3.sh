#!/usr/bin/env bash
# RL search round 3 — shaped reward + search2 structural fixes.
#
# Root cause of all prior failures:
#   Search1: shaped reward BUT bad config (curriculum, 4 epochs, 256 games → noisy advantages)
#   Search2: good config (no curriculum, 1 epoch, 512 games) BUT terminal reward → credit assignment failure
#
# The ONE untested combination: shaped reward (+1/−1/+5/−2) WITH search2's good config.
# All 3 experiments keep: no curriculum, ppo_epochs=1, games=512, ppo_minibatch=256
#
# Experiments (all 6k steps, sequential):
#   exp_rl3_a: shaped + lr=3e-6 + aux=0.10    — anchor (the untried combo)
#   exp_rl3_e: shaped + targeted sampling      — oversample j/k/w/z to fix letter bias
#   exp_rl3_f: proportional reward             — reward ∝ letters_revealed / word_len
#
# Rationale: 3 distinct hypotheses. lr/aux ablations (b/c/d) are premature until
# we know shaped reward works at all — run those in search4 if (a) succeeds.
#
# Total wall time: ~3 experiments × ~80 min = ~4 hours
#
# Abort logic:
#   After exp_rl3_a step 500: warn if val < 62%, but continue all 3 for data.
#   Decision tree after results:
#     a succeeds → run lr/aux sweep (search4)
#     a fails, e/f succeed → targeted sampling / proportional is key signal
#     all fail → algorithm bottleneck → add value head + GRPO (search4)
#
# Usage:
#   bash scripts/run_rl_search3.sh           # foreground
#   nohup bash scripts/run_rl_search3.sh &   # background, survive terminal close

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/rl_search3_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "rl_search3_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

INIT_CKPT="logs/sup_search4_20260518_1149/exp_k_250k_curr_4x2x/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════════════"
_log "  RL search3 run: ${SEARCH_ID}"
_log "  Init: exp_k 250k (search4) → test=66.20%  val=63.05%"
_log "  Hypothesis: shaped reward + search2 config = the untried combo"
_log "  Search1 had shaped reward BUT: curriculum+4epochs+256games→noisy"
_log "  Search2 fixed config BUT: terminal reward→credit assignment fail"
_log "  ─────────────────────────────────────────────────────────────"
_log "  Fixed across all: no-curriculum, ppo_epochs=1, games=512, mb=256"
_log "  ─────────────────────────────────────────────────────────────"
_log "  exp_rl3_a: shaped  lr=3e-6  aux=0.10  (anchor — untried combo)  6k steps"
_log "  exp_rl3_e: shaped  targeted lr=3e-6 aux=0.10 (20% j/k/w/z)   6k steps"
_log "  exp_rl3_f: proportional reward lr=3e-6 aux=0.10               6k steps"
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
        --eval-every      100             \
        --demo-every      500             \
        --log-every       20              \
        --val-subset      2000            \
        --ckpt-dir        "$ckpt_dir"     \
        --log-file        "${exp_dir}/rl.log" \
        "${extra_flags[@]}"
        # NO --curriculum, NO --terminal-reward

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
        _log "  best val: ${val_pct}%  ← rl_best.pt saved ✓"
    fi
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"

    # evaluate best checkpoint (rl_best.pt if saved, else supervised init)
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

# ── Abort check helper ────────────────────────────────────────────────────────
# Reads val_win_rate at step 500 from rl.log, warns if < 62% (does not abort)
check_early_val() {
    local log_file="$1"
    local step500_val
    step500_val=$(grep "eval ppo step=500" "$log_file" \
                  | grep -oP 'val_win_rate=\K[0-9.]+' | tail -1 || echo "")
    if [[ -z "$step500_val" ]]; then
        _log "  ⚠ step-500 eval not found in log — check manually"
        return
    fi
    local threshold=62
    if (( $(echo "$step500_val < $threshold" | bc -l) )); then
        _log "  ⚠ WARNING: step-500 val=${step500_val}% < ${threshold}%"
        _log "  ⚠ Shaped reward may also be failing. Continuing remaining experiments"
        _log "  ⚠ for data, but consider killing and redesigning RL approach."
    else
        _log "  ✓ step-500 val=${step500_val}% ≥ ${threshold}% — looks healthy"
    fi
}

# ── Run experiments sequentially ─────────────────────────────────────────────

run_exp "exp_rl3_a_shaped_anchor"  \
    --lr 3e-6  --aux-coef 0.10  --win-bonus 5.0 --loss-penalty -2.0
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"
check_early_val "${SEARCH_DIR}/exp_rl3_a_shaped_anchor/rl.log"

run_exp "exp_rl3_e_shaped_targeted"  \
    --lr 3e-6  --aux-coef 0.10  --win-bonus 5.0 --loss-penalty -2.0  \
    --targeted-sampling
VAL_E="$LAST_VAL"; TEST_E="$LAST_TEST"

run_exp "exp_rl3_f_proportional"  \
    --lr 3e-6  --aux-coef 0.10  \
    --proportional-reward
VAL_F="$LAST_VAL"; TEST_F="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════════════"
_log "  RL SEARCH3 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════════════"
_log "  supervised init (exp_k)                 val=63.05%  test=66.20%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  experiment                              val%        test%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  exp_rl3_a shaped lr=3e-6  aux=0.10     ${VAL_A}%      ${TEST_A}%"
_log "  exp_rl3_e shaped targeted lr=3e-6      ${VAL_E}%      ${TEST_E}%"
_log "  exp_rl3_f proportional    lr=3e-6      ${VAL_F}%      ${TEST_F}%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "  next: if (a) succeeds → search4 lr/aux sweep"
_log "        if all fail    → add value head + GRPO (search4)"
_log "═══════════════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
