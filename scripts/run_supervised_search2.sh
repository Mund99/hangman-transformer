#!/usr/bin/env bash
# Second round of supervised search — fills the 2×2 factorial:
#
#                     normal curr (4×/2×)   strong curr (8×/3×)
#   75k steps         exp_c ✓ done          exp_e  ← new
#   100k steps        exp_d  ← new          exp_f  ← new
#
# exp_d: 100k + curriculum(4×/2×)   — does more steps keep improving?
# exp_e:  75k + curriculum(8×/3×)   — stronger short focus, same duration
# exp_f: 100k + curriculum(8×/3×)   — combine both levers
#
# Usage:
#   bash scripts/run_supervised_search2.sh
#   bash scripts/run_supervised_search2.sh &   # background (recommended)

set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SEARCH_ID=$(date +%Y%m%d_%H%M)
SEARCH_DIR="logs/sup_search2_${SEARCH_ID}"
mkdir -p "$SEARCH_DIR"

ln -sfn "sup_search2_${SEARCH_ID}" logs/latest

SUMMARY="${SEARCH_DIR}/summary.log"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_log "═══════════════════════════════════════════════════════"
_log "  Supervised search2 run: ${SEARCH_ID}"
_log "  exp_d: 100k + curriculum(4×/2×)"
_log "  exp_e:  75k + curriculum(8×/3×)"
_log "  exp_f: 100k + curriculum(8×/3×)"
_log "  baseline: exp_c 75k+curr(4×/2×) → val=60.25%  test=62.76%"
_log "═══════════════════════════════════════════════════════"

# ── helper: run one experiment ──────────────────────────────────────────────
run_exp() {
    local name="$1"
    local steps="$2"
    local short_w="$3"
    local medium_w="$4"

    local exp_dir="${SEARCH_DIR}/${name}"
    local ckpt_dir="${exp_dir}/checkpoints"
    mkdir -p "$exp_dir" "$ckpt_dir"

    _log ""
    _log "▶ Starting ${name}  (steps=${steps}  short_weight=${short_w}  medium_weight=${medium_w})"
    echo "started: $(date)" > "${exp_dir}/run_info.txt"

    python scripts/train_supervised.py \
        --steps          "$steps"   \
        --bs             512        \
        --lr             3e-4       \
        --d-model        256        \
        --n-heads        4          \
        --n-layers       4          \
        --dim-ff         1024       \
        --log-every      200        \
        --eval-every     2000       \
        --demo-every     1000       \
        --val-subset     2000       \
        --workers        4          \
        --length-curriculum         \
        --short-weight   "$short_w" \
        --medium-weight  "$medium_w" \
        --ckpt-dir       "$ckpt_dir" \
        --log-file       "${exp_dir}/supervised.log"

    echo "train_done: $(date)" >> "${exp_dir}/run_info.txt"
    _log "✓ Training done: ${name}"

    local val_pct
    val_pct=$(grep "saved checkpoints/supervised_best.pt" "${exp_dir}/supervised.log" \
              | tail -1 | grep -oP 'val=\K[0-9.]+' || echo "n/a")
    echo "best_val: ${val_pct}%" >> "${exp_dir}/run_info.txt"
    _log "  best val: ${val_pct}%"

    local sup_ckpt="${ckpt_dir}/supervised_best.pt"
    if [[ -f "$sup_ckpt" ]]; then
        _log "  Running test evaluation…"
        python scripts/evaluate.py \
            --ckpt     "$sup_ckpt" \
            --words    data/test_words.txt \
            --log-file "${exp_dir}/eval_test.log"

        local test_pct
        test_pct=$(grep -oP 'TEST WIN RATE\s+:\s+\K[0-9.]+' "${exp_dir}/eval_test.log" \
                   | tail -1 || echo "n/a")
        echo "test_win_rate: ${test_pct}%" >> "${exp_dir}/run_info.txt"
        echo "eval_done: $(date)" >> "${exp_dir}/run_info.txt"
        _log "  test win-rate: ${test_pct}%"
    else
        _log "  WARNING: no checkpoint found at ${sup_ckpt}"
        test_pct="n/a"
    fi

    LAST_VAL="$val_pct"
    LAST_TEST="$test_pct"
}

# ── Run experiments sequentially ─────────────────────────────────────────────

run_exp "exp_d_100k_curr_4x2x"  100000  4.0  2.0
VAL_D="$LAST_VAL"; TEST_D="$LAST_TEST"

run_exp "exp_e_75k_curr_8x3x"    75000  8.0  3.0
VAL_E="$LAST_VAL"; TEST_E="$LAST_TEST"

run_exp "exp_f_100k_curr_8x3x"  100000  8.0  3.0
VAL_F="$LAST_VAL"; TEST_F="$LAST_TEST"

# ── Summary ───────────────────────────────────────────────────────────────────
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  SEARCH2 COMPLETE — ${SEARCH_ID}"
_log "═══════════════════════════════════════════════════════"
_log "  experiment                  val%       test%"
_log "  ─────────────────────────────────────────────────────"
_log "  baseline exp_c 75k(4×/2×)   60.25%     62.76%"
_log "  exp_d 100k curr(4×/2×)       ${VAL_D}%     ${TEST_D}%"
_log "  exp_e  75k curr(8×/3×)       ${VAL_E}%     ${TEST_E}%"
_log "  exp_f 100k curr(8×/3×)       ${VAL_F}%     ${TEST_F}%"
_log "  ─────────────────────────────────────────────────────"
_log "  target: ≥70% test win-rate"
_log "═══════════════════════════════════════════════════════"
_log "  Results in: ${SEARCH_DIR}/"
