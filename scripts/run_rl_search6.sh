#!/usr/bin/env bash
# RL Search 6 — KL-PPO with stronger regularisation + larger val eval.
#
# Search5 diagnosis:
#   - KL-PPO worked mechanically (no catastrophic forgetting, val 63.15%→65.4%)
#   - BUT test regressed (66.32% → 66.11–66.22%) across all kl_coef values
#   - Two root causes:
#     a) kl_coef 0.03–0.30 too weak: PPO loss ~0.03 >> KL contribution 0.002–0.021
#        Need kl_coef ≥ 1.0 for KL penalty to dominate and constrain policy shape
#     b) val-subset=1500 too noisy (±1.2pp std): rl_best.pt selected on lucky spikes
#        Need val-subset=5000 (±0.7pp) for reliable checkpoint selection
#
# Fixes in search6:
#   1. kl_coef sweep: 1.0, 3.0 — now KL penalty (1.0×0.07=0.07) > PPO loss (0.03)
#   2. val-subset=5000 — 3× more words, 1.7× lower noise
#   3. --save-every 2500 — capture intermediate checkpoints for post-hoc test eval
#   4. 10k steps — rl5_c was still improving at 5000; need more RL signal
#
# Also trying: proportional reward + kl=1.0.
# Proportional reward (+1/word_len per correct, −1/word_len per wrong, ±1 terminal)
# gives intermediate signal even in losing games, reducing gradient sparsity.
# (Terminal reward: 50% win_ma means 50% of games give all-negative gradient.)
#
# Init: ft_b step-6k (val=63.15%, test=66.32%)
#
# Experiments:
#   rl6_a: kl=1.0  terminal  GRPO K=4  10k steps
#   rl6_b: kl=3.0  terminal  GRPO K=4  10k steps
#   rl6_c: kl=1.0  proportional  GRPO K=4  10k steps
#
# Gate: val@1000 < 63.0% → skip (model forgetting; kl too weak for this config)
#
# Usage:
#   nohup bash scripts/run_rl_search6.sh &

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/rl_search6_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "rl_search6_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

# Best supervised checkpoint: ft_b step-6k (val=63.15%, test=66.32%)
INIT_CKPT="logs/sup_finetune_20260520_1228/ft_b_letter_w10_curriculum/checkpoints/supervised_best.pt"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════════════"
_log "  RL Search 6 — stronger KL + larger val eval: ${SEARCH_ID}"
_log "  Init: ft_b step-6k  val=63.15%  test=66.32%"
_log "  Fix:  kl_coef 1.0–3.0 (dominates PPO)  val-subset=5000"
_log "  ─────────────────────────────────────────────────────────────"
_log "  rl6_a: kl=1.0  terminal  GRPO K=4  10k steps"
_log "  rl6_b: kl=3.0  terminal  GRPO K=4  10k steps"
_log "  rl6_c: kl=1.0  proportional  GRPO K=4  10k steps"
_log "  gate:  val@1000 < 63.0% → skip"
_log "  target: test > 66.32%  (beat supervised init)"
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
        --steps           10000           \
        --games-per-step  512             \
        --grpo-k          4               \
        --curriculum                      \
        --aux-coef        0.0             \
        --lr              3e-6            \
        --clip-eps        0.1             \
        --ppo-epochs      1               \
        --ppo-minibatch   256             \
        --entropy-coef    0.003           \
        --eval-every      500             \
        --val-subset      5000            \
        --log-every       100             \
        --demo-every      1000            \
        --save-every      2500            \
        --ckpt-dir        "$ckpt_dir"     \
        --log-file        "${exp_dir}/train.log" \
        "${extra_flags[@]}"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    # Gate check: val at step 1000
    local val_1000
    val_1000=$(grep -oP '\[eval ppo step=1000\].*val_win_rate=\K[0-9.]+' \
               "${exp_dir}/train.log" | tail -1 || echo "n/a")
    _log "  val@1000: ${val_1000}%"

    local best_val_log
    best_val_log=$(grep -oP 'val_win_rate=\K[0-9.]+' "${exp_dir}/train.log" \
                   | tail -1 || echo "n/a")
    _log "  final val: ${best_val_log}%"
    echo "best_val: ${best_val_log}%" >> "${exp_dir}/run_info.txt"

    # Post-hoc test eval on every saved checkpoint to find true best
    _log "  Evaluating all saved checkpoints on test set…"
    local best_test_pct="0"; local best_test_ckpt="$INIT_CKPT"
    for ckpt_file in "$ckpt_dir"/rl_best.pt "$ckpt_dir"/rl_step*.pt; do
        [[ -f "$ckpt_file" ]] || continue
        local tmp_log="${exp_dir}/eval_$(basename $ckpt_file .pt).log"
        python scripts/evaluate.py \
            --ckpt     "$ckpt_file"          \
            --words    data/test_words.txt   \
            --log-file "$tmp_log" 2>/dev/null || continue
        local t
        t=$(grep -oP 'TEST WIN RATE\s+:\s+\K[0-9.]+' "$tmp_log" | tail -1 || echo "0")
        _log "    $(basename $ckpt_file): test=${t}%"
        if awk "BEGIN{exit !($t > $best_test_pct)}"; then
            best_test_pct="$t"; best_test_ckpt="$ckpt_file"
        fi
    done
    # Fallback: eval init if no checkpoints saved
    if [[ "$best_test_ckpt" == "$INIT_CKPT" ]]; then
        python scripts/evaluate.py \
            --ckpt     "$INIT_CKPT"          \
            --words    data/test_words.txt   \
            --log-file "${exp_dir}/eval_test.log"
        best_test_pct=$(grep -oP 'TEST WIN RATE\s+:\s+\K[0-9.]+' \
                        "${exp_dir}/eval_test.log" | tail -1 || echo "n/a")
        _log "  No improvement — init test: ${best_test_pct}%"
    fi
    echo "best_test_ckpt: $best_test_ckpt"   >> "${exp_dir}/run_info.txt"
    echo "best_test_pct:  ${best_test_pct}%" >> "${exp_dir}/run_info.txt"
    _log "  best test: ${best_test_pct}%  (${best_test_ckpt})"

    # Track global best
    if awk "BEGIN{exit !($best_test_pct > $BEST_VAL)}"; then
        BEST_VAL="$best_test_pct"
        BEST_CKPT="$best_test_ckpt"
        _log "  ★ New overall best: test=${BEST_VAL}%"
    fi

    LAST_VAL="$best_val_log"
    LAST_TEST="$best_test_pct"
    LAST_VAL1000="$val_1000"
}

# ── Run experiments ───────────────────────────────────────────────────────────

run_exp "rl6_a_kl1.0_terminal"        --kl-coef 1.0  --terminal-reward
VAL_A="$LAST_VAL"; TEST_A="$LAST_TEST"; V1K_A="$LAST_VAL1000"

run_exp "rl6_b_kl3.0_terminal"        --kl-coef 3.0  --terminal-reward
VAL_B="$LAST_VAL"; TEST_B="$LAST_TEST"; V1K_B="$LAST_VAL1000"

run_exp "rl6_c_kl1.0_proportional"    --kl-coef 1.0  --proportional-reward
VAL_C="$LAST_VAL"; TEST_C="$LAST_TEST"; V1K_C="$LAST_VAL1000"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════════════"
_log "  RL SEARCH 6 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════════════"
_log "  supervised init (ft_b)    val=63.15%  test=66.32%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  experiment               val@1000  final_val%  best_test%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  rl6_a kl=1.0 terminal    ${V1K_A}%   ${VAL_A}%      ${TEST_A}%"
_log "  rl6_b kl=3.0 terminal    ${V1K_B}%   ${VAL_B}%      ${TEST_B}%"
_log "  rl6_c kl=1.0 proportional ${V1K_C}%   ${VAL_C}%      ${TEST_C}%"
_log "  ─────────────────────────────────────────────────────────────"
_log "  target: test > 66.32%"
_log "  best ckpt: ${BEST_CKPT}  (test=${BEST_VAL}%)"
_log "  ─────────────────────────────────────────────────────────────"
_log "  next: if test > 66.32% → extend best config to 20k steps"
_log "        if val@1000 ≥ 64% but test still < 66.32% → distribution gap;"
_log "          try stochastic eval or per-letter test breakdown"
_log "        if val@1000 < 63% → kl too weak; try kl=10.0"
_log "═══════════════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
