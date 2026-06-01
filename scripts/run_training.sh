#!/usr/bin/env bash
# Full training pipeline: Stage 1 (supervised) → Stage 2 (RL) → Evaluation.
#
# Each run gets its own timestamped directory under logs/:
#   logs/20260514_1106/
#     run.log              pipeline-level events
#     supervised.log       stage 1 output
#     rl.log               stage 2 output
#     eval.log             stage 3 evaluation output
#     run_info.txt         args + timestamps
#     checkpoints/
#       supervised_best.pt
#       rl_best.pt          (only if RL improved over supervised)
#   logs/latest  →  20260514_1106/   (symlink, always points to current run)
#
# Usage
# ─────
#   Foreground:
#     bash scripts/run_training.sh
#
#   Background (recommended):
#     bash scripts/run_training.sh &
#
#   Resume RL after supervised already finished (must pass run dir):
#     bash scripts/run_training.sh --rl-only --run-dir logs/20260514_1106
#
#   Supervised only:
#     bash scripts/run_training.sh --sup-only

set -euo pipefail
cd "$(dirname "$0")/.."

# ── environment ───────────────────────────────────────────────────────────────
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="$(pwd)"

# MPS tuning: disable conservative memory watermark so MPS uses unified memory
# freely rather than stalling under pressure (Apple Silicon only, ignored elsewhere)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ── parse flags ───────────────────────────────────────────────────────────────
RL_ONLY=false
SUP_ONLY=false
EXISTING_RUN_DIR=""
SUP_STEPS=50000
LENGTH_CURRICULUM=false
EARLY_GAME_BIAS=false
for arg in "$@"; do
    case $arg in
        --rl-only)             RL_ONLY=true  ;;
        --sup-only)            SUP_ONLY=true ;;
        --run-dir=*)           EXISTING_RUN_DIR="${arg#*=}" ;;
        --run-dir)             shift; EXISTING_RUN_DIR="$1" ;;
        --sup-steps=*)         SUP_STEPS="${arg#*=}" ;;
        --sup-steps)           shift; SUP_STEPS="$1" ;;
        --length-curriculum)   LENGTH_CURRICULUM=true ;;
        --early-game-bias)     EARLY_GAME_BIAS=true ;;
    esac
done

# ── create run directory ──────────────────────────────────────────────────────
if [[ -n "$EXISTING_RUN_DIR" ]]; then
    RUN_DIR="$EXISTING_RUN_DIR"
    RUN_ID="$(basename "$RUN_DIR")"
else
    RUN_ID=$(date +%Y%m%d_%H%M)
    RUN_DIR="logs/${RUN_ID}"
fi

CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

# latest symlink always points to the active run
ln -sfn "$RUN_ID" logs/latest

RUN_LOG="${RUN_DIR}/run.log"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

_log "═══════════════════════════════════════════════════"
_log "  Hangman RL — training run ${RUN_ID}"
_log "  rl-only=${RL_ONLY}  sup-only=${SUP_ONLY}"
_log "═══════════════════════════════════════════════════"

# ── write run_info.txt ────────────────────────────────────────────────────────
cat > "${RUN_DIR}/run_info.txt" <<EOF
run_id    : ${RUN_ID}
started   : $(date)
rl_only   : ${RL_ONLY}
sup_only  : ${SUP_ONLY}
python    : $(python --version 2>&1)
device    : auto (MPS → CUDA → CPU)
EOF

SUP_CKPT="${CKPT_DIR}/supervised_best.pt"
RL_CKPT="${CKPT_DIR}/rl_best.pt"

# ── Stage 1: supervised pretraining ──────────────────────────────────────────
if [[ "$RL_ONLY" == "false" ]]; then
    _log "Stage 1 started — supervised pretraining"
    echo "stage_1_start: $(date)" >> "${RUN_DIR}/run_info.txt"

    EXTRA_SUP_FLAGS=""
    [[ "$LENGTH_CURRICULUM" == "true" ]] && EXTRA_SUP_FLAGS="$EXTRA_SUP_FLAGS --length-curriculum"
    [[ "$EARLY_GAME_BIAS"   == "true" ]] && EXTRA_SUP_FLAGS="$EXTRA_SUP_FLAGS --early-game-bias"

    python scripts/train_supervised.py \
        --steps      "$SUP_STEPS" \
        --bs         512   \
        --lr         3e-4  \
        --d-model    256   \
        --n-heads    4     \
        --n-layers   4     \
        --dim-ff     1024  \
        --log-every  200   \
        --eval-every 2000  \
        --demo-every 500   \
        --val-subset 2000  \
        --workers    4     \
        --ckpt-dir   "$CKPT_DIR" \
        --log-file   "${RUN_DIR}/supervised.log" \
        $EXTRA_SUP_FLAGS

    echo "stage_1_end: $(date)" >> "${RUN_DIR}/run_info.txt"
    _log "Stage 1 complete"
fi

# ── Stage 2: RL fine-tuning ───────────────────────────────────────────────────
if [[ "$SUP_ONLY" == "false" ]]; then
    if [[ ! -f "$SUP_CKPT" ]]; then
        _log "ERROR: ${SUP_CKPT} not found — run Stage 1 first"
        exit 1
    fi

    _log "Stage 2 started — RL fine-tuning"
    echo "stage_2_start: $(date)" >> "${RUN_DIR}/run_info.txt"

    python scripts/train_rl.py \
        --init           "$SUP_CKPT" \
        --steps          8000  \
        --games-per-step 512   \
        --lr             1e-5  \
        --clip-eps       0.2   \
        --ppo-epochs     4     \
        --entropy-coef   0.003 \
        --aux-coef       0.1   \
        --curriculum           \
        --eval-every     100   \
        --demo-every     100   \
        --log-every      20    \
        --ckpt-dir       "$CKPT_DIR" \
        --log-file       "${RUN_DIR}/rl.log"

    echo "stage_2_end: $(date)" >> "${RUN_DIR}/run_info.txt"
    _log "Stage 2 complete"
fi

# ── Stage 3: evaluation on held-out test set ─────────────────────────────────
if [[ "$SUP_ONLY" == "false" ]]; then
    # Prefer RL checkpoint; fall back to supervised if RL never improved
    if [[ -f "$RL_CKPT" ]]; then
        EVAL_CKPT="$RL_CKPT"
        _log "Stage 3 — evaluating rl_best.pt on test set"
    elif [[ -f "$SUP_CKPT" ]]; then
        EVAL_CKPT="$SUP_CKPT"
        _log "Stage 3 — rl_best.pt not saved (RL did not improve); evaluating supervised_best.pt on test set"
    else
        _log "Stage 3 — no checkpoint found, skipping evaluation"
        EVAL_CKPT=""
    fi

    if [[ -n "$EVAL_CKPT" ]]; then
        echo "stage_3_start: $(date)" >> "${RUN_DIR}/run_info.txt"

        python scripts/evaluate.py \
            --ckpt     "$EVAL_CKPT" \
            --words    data/test_words.txt \
            --log-file "${RUN_DIR}/eval.log"

        echo "stage_3_end: $(date)" >> "${RUN_DIR}/run_info.txt"
        _log "Stage 3 complete — see ${RUN_DIR}/eval.log"
    fi
fi

_log "All training complete — results in ${RUN_DIR}/"
