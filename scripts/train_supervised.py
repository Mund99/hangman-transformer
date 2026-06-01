"""
Stage 1: supervised pretraining.

For each training word we sample a random hangman state and train the model
to predict (26-way multi-label BCE) which letters appear in the true word.
Loss is computed only on letters not yet guessed in that state.

Periodically the model plays full simulated games on the validation set.
Best-by-val-win-rate checkpoint is saved to checkpoints/supervised_best.pt.

Usage (run from project root):
    python scripts/train_supervised.py
    python scripts/train_supervised.py --steps 30000 --bs 512 --lr 3e-4
    python scripts/train_supervised.py --log-file logs/supervised.log
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from hangman.dataset import HangmanStateDataset, load_word_list
from hangman.model import HangmanPolicy, ModelConfig, count_parameters
from hangman.eval import evaluate_win_rate
from utils import pick_device, setup_logging, demo_game


def train(args, logger):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    demo_rng = random.Random()  # seeded from os.urandom — different words each run

    device = pick_device(args.device)
    logger.info(f"[device] {device}")

    train_words = load_word_list(args.train_file)
    val_words   = load_word_list(args.val_file)
    logger.info(f"[data] train={len(train_words):,}  val={len(val_words):,}")
    logger.info(f"[config] length_curriculum={args.length_curriculum}  "
                f"short_weight={args.short_weight}  medium_weight={args.medium_weight}  "
                f"rare_weight={args.rare_weight}  rare_letter_weight={args.rare_letter_weight}  "
                f"early_game_bias={args.early_game_bias}  steps={args.steps}")

    ds = HangmanStateDataset(train_words, seed=args.seed,
                             early_game_bias=args.early_game_bias)

    _RARE = set("jkwzqx")

    if args.length_curriculum or args.rare_weight > 1.0:
        sw, mw = args.short_weight, args.medium_weight
        rw     = args.rare_weight

        def _word_weight(w: str) -> float:
            # Length curriculum weight
            n = len(w)
            length_w = sw if n <= 6 else (mw if n <= 9 else 1.0)
            # Rare-letter multiplier: any word containing j/k/w/z/q/x
            rare_w = rw if any(c in _RARE for c in w) else 1.0
            return length_w * rare_w

        weights = [_word_weight(w) for w in train_words]
        n_rare  = sum(1 for w in train_words if any(c in _RARE for c in w))
        logger.info(f"[sampler] WeightedRandomSampler  rare_weight={rw}  "
                    f"rare_words={n_rare:,}/{len(train_words):,} "
                    f"({n_rare/len(train_words)*100:.1f}%)")
        sampler = WeightedRandomSampler(weights, num_samples=len(train_words),
                                        replacement=True)
        loader = DataLoader(
            ds,
            batch_size=args.bs,
            sampler=sampler,
            num_workers=args.workers,
            drop_last=True,
            persistent_workers=args.workers > 0,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            persistent_workers=args.workers > 0,
        )

    if args.init:
        ckpt_data = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg   = ModelConfig(**ckpt_data["cfg"])
        model = HangmanPolicy(cfg).to(device)
        model.load_state_dict(ckpt_data["model_state"])
        best_val = ckpt_data.get("val_win_rate", -1.0)
        logger.info(f"[init] loaded checkpoint  val={best_val*100:.2f}%  ({args.init})")
    else:
        cfg = ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
        )
        model = HangmanPolicy(cfg).to(device)
    logger.info(f"[model] params={count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    # Per-letter BCE weight vector — upweights rare letters (j/k/w/z/q/x) in the loss.
    # Shape [26], broadcasts over [batch, 26]. All ones = uniform (default).
    _RARE_IDX = [ord(c) - ord('a') for c in "jkwzqx"]
    letter_w_vals = [args.rare_letter_weight if i in _RARE_IDX else 1.0 for i in range(26)]
    letter_w = torch.tensor(letter_w_vals, dtype=torch.float32, device=device)
    if args.rare_letter_weight > 1.0:
        logger.info(f"[loss] rare_letter_weight={args.rare_letter_weight}  "
                    f"letters upweighted: j k w z q x")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not args.init:
        best_val = -1.0   # will be set by first eval; --init sets it above
    step = 0
    t_start = time.time()
    val_eval = val_words[: args.val_subset] if args.val_subset > 0 else val_words

    it = iter(loader)
    running_loss = 0.0
    running_n = 0

    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        tok = batch["token_ids"].to(device, non_blocking=True)
        am  = batch["attn_mask"].to(device, non_blocking=True)
        gv  = batch["guessed_vec"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        lm  = batch["loss_mask"].to(device, non_blocking=True)

        logits = model(tok, am, gv)
        bce    = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        # Optional per-letter loss weighting: upweight rare letters in BCE.
        # letter_w shape [26] broadcasts over [batch, 26].
        # Denominator uses the same weights so scale stays correct.
        loss   = (bce * lm * letter_w).sum() / (lm * letter_w).sum().clamp(min=1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item() * lm.sum().item()
        running_n    += lm.sum().item()
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(f"[step {step:>6}/{args.steps}]  "
                  f"loss={running_loss / max(1.0, running_n):.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"elapsed={elapsed:6.1f}s")
            running_loss = 0.0
            running_n    = 0

        if args.demo_every > 0 and step % args.demo_every == 0:
            word = demo_rng.choice(val_words)
            logger.info(f"[demo step={step}] ── greedy game on '{word}' ──────────────")
            demo_game(model, word, device, logger)

        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__,
                 "step": step, "val_win_rate": best_val},
                ckpt_dir / f"supervised_step{step}.pt",
            )
            logger.info(f"  → periodic checkpoint saved: supervised_step{step}.pt")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_win_rate(model, val_eval, device)
            logger.info(f"[eval  step={step}]  val_win_rate={metrics['win_rate']*100:.2f}%  "
                  f"(n={metrics['n_games']:,}, avg_wrong={metrics['avg_wrong_per_game']:.2f})")
            if metrics["win_rate"] > best_val:
                best_val = metrics["win_rate"]
                torch.save(
                    {"model_state": model.state_dict(), "cfg": cfg.__dict__,
                     "step": step, "val_win_rate": best_val},
                    ckpt_dir / "supervised_best.pt",
                )
                logger.info(f"  → saved checkpoints/supervised_best.pt  (val={best_val*100:.2f}%)")

    logger.info(f"\n[done] best val win-rate: {best_val*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="data/train_words.txt",
                    help="Path to training word list")
    ap.add_argument("--val-file",   default="data/val_words.txt",
                    help="Path to validation word list")
    ap.add_argument("--steps",      type=int,   default=30_000)
    ap.add_argument("--warmup",     type=int,   default=1_000)
    ap.add_argument("--bs",         type=int,   default=512)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--d-model",    type=int,   default=128)
    ap.add_argument("--n-heads",    type=int,   default=4)
    ap.add_argument("--n-layers",   type=int,   default=4)
    ap.add_argument("--dim-ff",     type=int,   default=512)
    ap.add_argument("--dropout",    type=float, default=0.1)
    ap.add_argument("--workers",    type=int,   default=0)
    ap.add_argument("--device",     default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed",       type=int,   default=1337)
    ap.add_argument("--log-every",  type=int,   default=200)
    ap.add_argument("--eval-every", type=int,   default=2_000)
    ap.add_argument("--val-subset", type=int,   default=2_000,
                    help="Eval on the first N val words for speed; 0 = all")
    ap.add_argument("--demo-every", type=int,   default=1_000,
                    help="Play and log a demo game every N steps (0 = off)")
    ap.add_argument("--ckpt-dir",          default="checkpoints",
                    help="Directory to save checkpoints (default: checkpoints)")
    ap.add_argument("--log-file",          default=None,
                    help="Log file path (e.g., logs/supervised.log)")
    ap.add_argument("--init",              default=None,
                    help="Resume fine-tuning from this checkpoint (.pt). "
                         "Architecture cfg is loaded from the checkpoint.")
    ap.add_argument("--rare-letter-weight", type=float, default=1.0,
                    help="Multiply BCE loss weight by N for letters j/k/w/z/q/x. "
                         "Upweights rare-letter signal without disrupting word distribution. "
                         "(default: 1.0 = uniform)")
    ap.add_argument("--rare-weight",       type=float, default=1.0,
                    help="Extra sampling weight for words containing j/k/w/z/q/x. "
                         "E.g. 10.0 → rare-letter words appear 10× more often. "
                         "Stacks multiplicatively with --length-curriculum weights.")
    ap.add_argument("--length-curriculum", action="store_true",
                    help="Use WeightedRandomSampler to upsample short words")
    ap.add_argument("--short-weight",      type=float, default=4.0,
                    help="Curriculum weight for words len≤6 (default 4.0)")
    ap.add_argument("--medium-weight",     type=float, default=2.0,
                    help="Curriculum weight for words len 7-9 (default 2.0)")
    ap.add_argument("--early-game-bias",   action="store_true",
                    help="Bias state sampling toward early-game (fewer revealed letters) for short/medium words")
    ap.add_argument("--save-every",        type=int, default=0,
                    help="Save a periodic checkpoint every N steps (0 = off)")
    args = ap.parse_args()

    logger = setup_logging(args.log_file)
    train(args, logger)


if __name__ == "__main__":
    main()
