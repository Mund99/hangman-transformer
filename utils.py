"""Shared utilities for training scripts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    also_console: bool = True,
) -> logging.Logger:
    """
    Setup logging to file and optionally console.

    Returns a logger that writes to both file (if provided) and console.

    Usage:
        logger = setup_logging("logs/train_supervised.log")
        logger.info("Training started")
    """
    logger = logging.getLogger("hangman_rl")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if also_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def demo_game(model, word: str, device: torch.device, logger) -> None:
    """Play one greedy game and log every guess. Shared by supervised and RL scripts."""
    from hangman.env import HangmanGame
    from hangman.vocab import MAX_WORD_LEN, encode_masked, encode_guessed

    model.eval()
    game = HangmanGame(word)
    logger.info(f"  [demo] word length: {len(word)}   {'_ ' * len(word)}")

    with torch.no_grad():
        while not game.done:
            ids, am = encode_masked(game.masked, MAX_WORD_LEN)
            gv      = encode_guessed(game.guessed)
            tok     = torch.from_numpy(ids[None]).to(device)
            atm     = torch.from_numpy(am[None]).to(device)
            gvt     = torch.from_numpy(gv[None]).to(device)

            logits = model(tok, atm, gvt)[0]
            logits = logits.masked_fill(torch.from_numpy(gv).bool().to(device), float("-inf"))
            probs  = torch.softmax(logits, dim=0)
            idx    = int(logits.argmax().item())
            letter = chr(ord("a") + idx)
            prob   = probs[idx].item()

            hit    = game.guess(letter)
            symbol = "✓" if hit else "✗"
            masked_display = " ".join(c if c != "_" else "_" for c in game.masked)
            logger.info(
                f"  [demo]   guess '{letter}' (p={prob:.0%}) {symbol}  "
                f"│ {masked_display}  │  lives: {game.lives_remaining}"
            )

    outcome = "WIN" if game.won else "LOSS"
    logger.info(f"  [demo] → {outcome}  (word: {game.word})")
    model.train()
