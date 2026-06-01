"""
Ensemble evaluation: route by word length.

  len ≤ split_len  →  short-word model  (dedicated, trained only on short words)
  len >  split_len  →  main model        (sup5_b, trained on full vocabulary)

Usage
-----
    # Default: split at len=6 (short model handles 4-6, main handles 7+)
    python scripts/evaluate_ensemble.py \
        --ckpt-main  logs/sup_search5_20260521_0116/sup5_b_d512_L6/checkpoints/supervised_best.pt \
        --ckpt-short logs/sup_search7_.../sw_a.../checkpoints/supervised_best.pt

    # Try different split lengths
    python scripts/evaluate_ensemble.py \
        --ckpt-main  ... --ckpt-short ... --split-len 7

    # Evaluate the short model alone on short words only (no ensemble)
    python scripts/evaluate_ensemble.py \
        --ckpt-short ... --short-only
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hangman.dataset import load_word_list
from hangman.eval import evaluate_with_wins
from hangman.model import HangmanPolicy, ModelConfig
from utils import pick_device


def load_model(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg   = ModelConfig(**ckpt["cfg"])
    model = HangmanPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def stratify_by_length(words, wins):
    by_len = defaultdict(lambda: [0, 0])
    for w, won in zip(words, wins):
        by_len[len(w)][0] += 1
        by_len[len(w)][1] += int(won)
    return {L: (n, k, k / max(1, n)) for L, (n, k) in sorted(by_len.items())}


def bar(r, width=20):
    return "█" * int(r * width) + "░" * (width - int(r * width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-main",  default=None, help="Main model checkpoint (full vocab)")
    ap.add_argument("--ckpt-short", required=True, help="Short-word model checkpoint")
    ap.add_argument("--words",      default="data/test_words.txt")
    ap.add_argument("--split-len",  type=int, default=6,
                    help="Words ≤ split_len use short model; > split_len use main (default: 6)")
    ap.add_argument("--short-only", action="store_true",
                    help="Evaluate short model only on words ≤ split_len (no ensemble)")
    ap.add_argument("--device",     default="auto", choices=["auto","mps","cuda","cpu"])
    ap.add_argument("--log-file",   default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[device]  {device}")

    # ── Load models ────────────────────────────────────────────────────────────
    short_model, short_ckpt = load_model(args.ckpt_short, device)
    print(f"[short]   {args.ckpt_short}  "
          f"step={short_ckpt.get('step','?')}  "
          f"val={short_ckpt.get('val_win_rate',0)*100:.2f}%")

    if args.ckpt_main and not args.short_only:
        main_model, main_ckpt = load_model(args.ckpt_main, device)
        print(f"[main]    {args.ckpt_main}  "
              f"step={main_ckpt.get('step','?')}  "
              f"val={main_ckpt.get('val_win_rate',0)*100:.2f}%")
    else:
        main_model = main_ckpt = None

    # ── Load words ─────────────────────────────────────────────────────────────
    all_words = load_word_list(args.words)
    print(f"[words]   {len(all_words):,} total")

    W = 62
    log_lines = []

    def _log(line=""):
        print(line)
        log_lines.append(line)

    if args.short_only or main_model is None:
        # ── Short model alone on short words ───────────────────────────────────
        short_words = [w for w in all_words if len(w) <= args.split_len]
        _log(f"\n{'='*W}")
        _log(f"  SHORT-WORD MODEL ONLY  (len ≤ {args.split_len})")
        _log(f"  Checkpoint : {args.ckpt_short}")
        _log(f"  Words      : {len(short_words):,} (of {len(all_words):,} total)")
        _log(f"{'='*W}")

        wins = evaluate_with_wins(short_model, short_words, device)
        wr   = sum(wins) / max(1, len(wins))
        _log(f"  WIN RATE (short only): {wr*100:.2f}%  ({sum(wins):,}/{len(wins):,})")

        by_len = stratify_by_length(short_words, wins)
        _log(f"\n  {'LEN':>4}  {'N':>6}  {'WINS':>5}  {'WIN%':>7}  BAR")
        _log(f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*20}")
        for L, (n, k, r) in by_len.items():
            flag = "  ◄ weak" if r < 0.40 else ""
            _log(f"  {L:>4}  {n:>6,}  {k:>5,}  {r*100:>6.1f}%  {bar(r)}{flag}")

    else:
        # ── Full ensemble evaluation ────────────────────────────────────────────
        short_words = [w for w in all_words if len(w) <= args.split_len]
        long_words  = [w for w in all_words if len(w) >  args.split_len]

        _log(f"\n{'='*W}")
        _log(f"  ENSEMBLE EVALUATION")
        _log(f"  Short model : {args.ckpt_short}")
        _log(f"  Main model  : {args.ckpt_main}")
        _log(f"  Split       : len ≤ {args.split_len} → short model  "
             f"({len(short_words):,} words)")
        _log(f"              : len > {args.split_len} → main model   "
             f"({len(long_words):,} words)")
        _log(f"{'='*W}")

        short_wins = evaluate_with_wins(short_model, short_words, device)
        long_wins  = evaluate_with_wins(main_model,  long_words,  device)

        # Reassemble in original order
        all_wins   = [False] * len(all_words)
        si = li = 0
        for idx, w in enumerate(all_words):
            if len(w) <= args.split_len:
                all_wins[idx] = short_wins[si]; si += 1
            else:
                all_wins[idx] = long_wins[li];  li += 1

        total_wr    = sum(all_wins)  / len(all_wins)
        short_wr    = sum(short_wins) / max(1, len(short_wins))
        long_wr     = sum(long_wins)  / max(1, len(long_wins))

        _log(f"\n  OVERALL WIN RATE : {total_wr*100:.2f}%  "
             f"({sum(all_wins):,}/{len(all_wins):,})")
        _log(f"  Short (≤{args.split_len})       : {short_wr*100:.2f}%  "
             f"({sum(short_wins):,}/{len(short_wins):,})")
        _log(f"  Long  (>{args.split_len})        : {long_wr*100:.2f}%  "
             f"({sum(long_wins):,}/{len(long_wins):,})")

        # Per-length table
        by_len = stratify_by_length(all_words, all_wins)
        _log(f"\n  {'LEN':>4}  {'N':>6}  {'WINS':>5}  {'WIN%':>7}  {'MODEL':^6}  BAR")
        _log(f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*20}")
        for L, (n, k, r) in by_len.items():
            model_tag = "SHORT" if L <= args.split_len else "MAIN "
            flag = "  ◄ weak" if r < 0.40 else ("  ✓" if r >= 0.70 else "")
            _log(f"  {L:>4}  {n:>6,}  {k:>5,}  {r*100:>6.1f}%  {model_tag}  {bar(r)}{flag}")

        _log(f"\n{'='*W}")

    # ── Save log ────────────────────────────────────────────────────────────────
    log_path = args.log_file
    if log_path is None:
        out_dir = Path(args.ckpt_short).parent.parent / "ensemble_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix  = "short_only" if (args.short_only or not main_model) else f"ensemble_split{args.split_len}"
        log_path = str(out_dir / f"{suffix}.log")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  [saved]  {log_path}")


if __name__ == "__main__":
    main()
