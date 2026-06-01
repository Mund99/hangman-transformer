"""
Final evaluation on held-out test words — the 70% gate.
Touch the test set ONLY here; never during training.

Usage (run from project root):
    python scripts/evaluate.py --ckpt checkpoints/rl_best.pt
    python scripts/evaluate.py --ckpt checkpoints/supervised_best.pt --n 2000
    python scripts/evaluate.py --ckpt checkpoints/rl_best.pt --verbose 5 --log-file logs/eval.log
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from hangman.dataset import load_word_list
from hangman.model import HangmanPolicy, ModelConfig
from hangman.vocab import MAX_WORD_LEN, encode_masked, encode_guessed
from hangman.env import MAX_WRONG
from utils import pick_device, setup_logging


def stratify_by_length(words: list[str], wins: list[bool]) -> dict:
    by_len: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for w, won in zip(words, wins):
        by_len[len(w)][0] += 1
        by_len[len(w)][1] += int(won)
    return {L: (n, k, k / max(1, n)) for L, (n, k) in sorted(by_len.items())}


def stratify_by_letter(words: list[str], wins: list[bool]) -> dict:
    by_letter: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for w, won in zip(words, wins):
        for ch in set(w):
            by_letter[ch][0] += 1
            by_letter[ch][1] += int(won)
    return {ch: (n, k, k / max(1, n)) for ch, (n, k) in sorted(by_letter.items())}


@torch.no_grad()
def evaluate_with_wins_verbose(
    model,
    words: list[str],
    device: torch.device,
    logger,
    verbose_n: int = 0,
    max_wrong: int = MAX_WRONG,
    batch_size: int = 512,
) -> list[bool]:
    """
    Evaluate and optionally show verbose output for first N games.
    Returns per-word win flags.
    """
    model.eval()
    all_wins = [False] * len(words)
    verbose_words = words[:verbose_n] if verbose_n > 0 else []

    for start in range(0, len(words), batch_size):
        batch = words[start : start + batch_size]
        revealed = [set() for _ in batch]
        wrong = [set() for _ in batch]
        done = [False] * len(batch)
        result = [False] * len(batch)

        for _step in range(26):
            active = [i for i, d in enumerate(done) if not d]
            if not active:
                break

            tok_list, am_list, gv_list = [], [], []
            for i in active:
                masked = "".join(c if c in revealed[i] else "_" for c in batch[i])
                ids, am = encode_masked(masked, MAX_WORD_LEN)
                gv = encode_guessed(revealed[i] | wrong[i])
                tok_list.append(ids)
                am_list.append(am)
                gv_list.append(gv)

            tok = torch.from_numpy(np.stack(tok_list)).to(device)
            am = torch.from_numpy(np.stack(am_list)).to(device)
            gv = torch.from_numpy(np.stack(gv_list)).to(device)

            logits = model(tok, am, gv)
            logits = logits.masked_fill(gv.bool(), float("-inf"))
            actions = logits.argmax(dim=-1).cpu().tolist()

            for slot, i in enumerate(active):
                letter = chr(ord("a") + actions[slot])
                w = batch[i]

                # Verbose logging for first N words
                if start + i < verbose_n:
                    global_idx = start + i
                    is_first_step = (len(revealed[i]) + len(wrong[i])) == 0
                    if is_first_step:
                        logger.info(f"\n=== Game {global_idx + 1}: {w} ===")

                if letter in w:
                    revealed[i].add(letter)
                    if start + i < verbose_n:
                        logger.info(f"  Guess '{letter}': ✓  →  {''.join(c if c in revealed[i] else '_' for c in w)}  lives={max_wrong - len(wrong[i])}")
                    if all(c in revealed[i] for c in w):
                        done[i] = True
                        result[i] = True
                        if start + i < verbose_n:
                            logger.info(f"  Result: 🎉 WIN!")
                else:
                    wrong[i].add(letter)
                    if start + i < verbose_n:
                        logger.info(f"  Guess '{letter}': ✗  →  {''.join(c if c in revealed[i] else '_' for c in w)}  lives={max_wrong - len(wrong[i])}")
                    if len(wrong[i]) >= max_wrong:
                        done[i] = True
                        result[i] = False
                        if start + i < verbose_n:
                            logger.info(f"  Result: 💀 LOSS! Word was: {w}")

        for j, i in enumerate(range(start, start + len(batch))):
            all_wins[i] = result[j]

    return all_wins


def _report(lines: list[str], log_file: str | None) -> None:
    """Write clean report lines to stdout and append to log file without timestamps."""
    text = "\n".join(lines)
    print(text)
    if log_file:
        with open(log_file, "a") as f:
            f.write("\n" + text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--words", default="data/test_words.txt")
    ap.add_argument("--n", type=int, default=0, help="Limit to first N words; 0=all")
    ap.add_argument("--verbose", type=int, default=0,
                    help="Show detailed play-by-play for first N games (default: 0)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--log-file", default=None,
                    help="Log file path (e.g., logs/eval.log)")
    args = ap.parse_args()

    logger = setup_logging(args.log_file)
    device = pick_device(args.device)
    logger.info(f"[device] {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"])
    model = HangmanPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"[load] {args.ckpt}  step={ckpt.get('step','?')}  val={ckpt.get('val_win_rate',0)*100:.2f}%")

    words = load_word_list(args.words)
    if args.n > 0:
        words = words[: args.n]
    logger.info(f"[eval] {len(words):,} words from {args.words}")

    if args.verbose > 0:
        verbose_n = min(args.verbose, len(words))
        logger.info(f"[verbose] showing first {verbose_n} games in detail")
    else:
        verbose_n = 0

    wins = evaluate_with_wins_verbose(model, words, device, logger, verbose_n=verbose_n)
    wr   = sum(wins) / len(wins)
    n_win, n_tot = sum(wins), len(wins)
    by_len    = stratify_by_length(words, wins)
    by_letter = stratify_by_letter(words, wins)

    # ── clean report (no timestamps) ─────────────────────────────────────────
    W = 62
    passed = wr >= 0.70
    verdict = "✅  PASSED 70% threshold" if passed else f"❌  Below 70% threshold ({wr*100:.2f}% < 70%)"

    lines = [
        "",
        "=" * W,
        f"  EVALUATION REPORT",
        f"  Checkpoint : {args.ckpt}",
        f"  Word list  : {args.words}  ({n_tot:,} words)",
        f"  Val win    : {ckpt.get('val_win_rate', 0)*100:.2f}%  (during training)",
        "=" * W,
        f"  TEST WIN RATE  :  {wr*100:.2f}%   ({n_win:,} / {n_tot:,})",
        f"  {verdict}",
        "=" * W,
        "",
        f"  {'LEN':>4}  {'WORDS':>6}  {'WINS':>6}  {'WIN%':>7}  {'BAR':<25}",
        f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*25}",
    ]
    for L, (n, k, r) in by_len.items():
        bar_len = int(r * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        flag = "  ◄ weak" if r < 0.40 else ("  ✓" if r >= 0.70 else "")
        lines.append(f"  {L:>4}  {n:>6,}  {k:>6,}  {r*100:>6.1f}%  {bar}{flag}")

    # ── per-letter section ────────────────────────────────────────────────────
    lines += [
        "",
        "  PER-LETTER WIN RATE  (words containing each letter)",
        f"  {'LTR':>4}  {'WORDS':>7}  {'WINS':>7}  {'WIN%':>7}  {'VS AVG':>9}  {'BAR':<25}",
        f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*25}",
    ]
    for ch, (n, k, r) in by_letter.items():
        diff = r - wr
        sign = "+" if diff >= 0 else ""
        bar_len = int(r * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        flag = "  ◄ BIAS" if diff < -0.10 else ("  ✓" if r >= 0.70 else "")
        lines.append(
            f"  '{ch}'   {n:>7,}  {k:>7,}  {r*100:>6.1f}%  {sign}{diff*100:>+6.1f}pp  {bar}{flag}"
        )

    # rare-letter sample failures
    rare_letters = [ch for ch, (n, k, r) in by_letter.items() if r - wr < -0.10]
    if rare_letters:
        lines += ["", "  RARE/BIASED LETTERS — sample failures (up to 15 words each)"]
        for ch in rare_letters:
            failed = [w for w, won in zip(words, wins) if ch in w and not won][:15]
            n, k, r = by_letter[ch]
            lines.append(f"  '{ch}'  {n} words  {r*100:.1f}%  →  {', '.join(failed)}")

    lines += ["", "=" * W, ""]
    _report(lines, args.log_file)


if __name__ == "__main__":
    main()
