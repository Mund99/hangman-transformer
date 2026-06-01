"""
Analyze per-letter bias in model failures.

For each letter a-z, computes:
  - how many test words contain that letter
  - win rate on words that contain that letter
  - fail rate (complement)

Usage:
    python scripts/analyze_letter_bias.py --ckpt <path> --words data/test_words.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hangman.model import HangmanPolicy, ModelConfig
from hangman.dataset import load_word_list
from hangman.eval import evaluate_with_wins
from utils import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  required=True)
    ap.add_argument("--words", default="data/test_words.txt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg  = ModelConfig(**ckpt["cfg"])
    model = HangmanPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    words = load_word_list(args.words)
    print(f"Evaluating {len(words):,} words on {device}…", flush=True)

    wins = evaluate_with_wins(model, words, device)

    # per-letter stats
    letter_words  = defaultdict(list)  # letter -> list of (word, won)
    for word, won in zip(words, wins):
        for ch in set(word):
            letter_words[ch].append(won)

    # overall
    overall = sum(wins) / len(wins) * 100

    print(f"\nOverall win rate: {overall:.2f}%  ({sum(wins)}/{len(wins)})\n")
    print(f"{'letter':>6}  {'words':>7}  {'wins':>7}  {'win%':>7}  {'vs overall':>10}  bar")
    print("─" * 72)

    for ch in sorted(letter_words.keys()):
        ws = letter_words[ch]
        n  = len(ws)
        k  = sum(ws)
        pct = k / n * 100
        diff = pct - overall
        bar_fill = int(pct / 4)
        bar = "█" * bar_fill + "░" * (25 - bar_fill)
        sign = "+" if diff >= 0 else ""
        print(f"  '{ch}'   {n:>7,}  {k:>7,}  {pct:>6.1f}%  {sign}{diff:>+7.1f}pp  {bar}")

    # rare-letter deep dive: words with z,x,q,j,w
    rare = list("zxqjw")
    print("\n── Rare-letter deep dive ─────────────────────────────────────────────")
    for ch in rare:
        ws_pairs = [(w, won) for w, won in zip(words, wins) if ch in w]
        n  = len(ws_pairs)
        k  = sum(won for _, won in ws_pairs)
        pct = k / max(1, n) * 100
        # show up to 10 failed words
        failed = [w for w, won in ws_pairs if not won][:20]
        print(f"\n  '{ch}'  {n} words  {k} wins  ({pct:.1f}%)")
        print(f"  sample failures: {', '.join(failed[:20]) or 'none'}")


if __name__ == "__main__":
    main()
