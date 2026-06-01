"""
Training data: word loading, state sampling, and the PyTorch Dataset.

A "state" is what the model sees during a game:
  - masked word  ("_ a _ p l e")
  - letters already guessed (correct AND wrong) — binary vector [26]

The target is the multi-label set of letters still hidden in the true word.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from hangman.vocab import (
    LETTERS,
    LETTER_TO_IDX,
    MAX_WORD_LEN,
    encode_masked,
    encode_guessed,
)
from hangman.env import make_masked


# ---------------------------------------------------------------------------
# Word loading
# ---------------------------------------------------------------------------

def load_word_list(path: str | Path) -> list[str]:
    return [w.strip() for w in Path(path).read_text().splitlines() if w.strip()]


# ---------------------------------------------------------------------------
# State sampling
# ---------------------------------------------------------------------------

@dataclass
class HangmanState:
    word: str
    masked: str
    guessed: set[str]
    revealed: set[str]
    wrong: set[str]

    def remaining_in_word(self) -> set[str]:
        return set(self.word) - self.revealed


def _geometric(rng: random.Random, p: float) -> int:
    """Number of failures before first success ~ Geometric(p)."""
    k = 0
    while rng.random() > p:
        k += 1
        if k > 50:
            return k
    return k


def sample_state(word: str, rng: random.Random,
                 p_reveal: float | None = None) -> HangmanState:
    """
    Sample a plausible mid-game hangman state from a target word.

    - k correctly-revealed letters.
      Default (p_reveal=None): uniform in [0, |unique|-1] — equal probability
      for any number of pre-revealed correct letters.
      When p_reveal is set: k ~ Geometric(p_reveal) capped at |unique|-1,
      biasing toward early-game states (fewer letters already revealed).
        p_reveal=0.7 → P(k=0)=70%, P(k=1)=21%, P(k=2)=6%  (short words)
        p_reveal=0.5 → P(k=0)=50%, P(k=1)=25%, P(k=2)=12% (medium words)
    - j wrong guesses, Geometric(0.35) capped at 5.
    """
    unique = list(set(word))
    rng.shuffle(unique)

    max_k = max(len(unique) - 1, 0)
    if max_k == 0:
        k = 0
    elif p_reveal is not None:
        k = min(max_k, _geometric(rng, p_reveal))
    else:
        k = rng.randint(0, max_k)
    revealed = set(unique[:k])

    not_in_word = [c for c in LETTERS if c not in set(word)]
    rng.shuffle(not_in_word)
    j = min(5, _geometric(rng, 0.35))
    wrong = set(not_in_word[:j])

    return HangmanState(
        word=word,
        masked=make_masked(word, revealed),
        guessed=revealed | wrong,
        revealed=revealed,
        wrong=wrong,
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def encode_target(state: HangmanState) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (target, loss_mask):
      target[i]    = 1 if letter i is anywhere in the word
      loss_mask[i] = 1 if letter i has not yet been guessed
                     (we only penalize predictions on unknown letters)
    """
    target = np.zeros(26, dtype=np.float32)
    for c in set(state.word):
        target[LETTER_TO_IDX[c]] = 1.0

    loss_mask = np.ones(26, dtype=np.float32)
    for c in state.guessed:
        loss_mask[LETTER_TO_IDX[c]] = 0.0

    return target, loss_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HangmanStateDataset(Dataset):
    """
    Each __getitem__ samples a fresh random state from a fixed word, so
    the same word index yields different training examples across epochs.

    Returns a dict of tensors:
      token_ids   : long  [max_len]
      attn_mask   : long  [max_len]
      guessed_vec : float [26]
      target      : float [26]
      loss_mask   : float [26]
    """

    def __init__(self, words: Sequence[str], max_len: int = MAX_WORD_LEN,
                 seed: int = 0, early_game_bias: bool = False):
        self.words = list(words)
        self.max_len = max_len
        self._seed = seed
        self._early_game_bias = early_game_bias

    def _p_reveal(self, word: str) -> float | None:
        """Return p_reveal for Geometric state sampling, or None for uniform."""
        if not self._early_game_bias:
            return None
        n = len(word)
        if n <= 6:
            return 0.7   # P(k=0)=70% P(k=1)=21% — mostly early-game
        if n <= 9:
            return 0.5   # P(k=0)=50% P(k=1)=25%
        return None      # long words: uniform (they're already learned well)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Mix a fresh random 64-bit value so the same word generates a different
        # masked state on every call (across epochs, workers, and shuffles).
        # Without this, a fixed seed per idx causes the model to see the same
        # state for each word every epoch — it memorizes rather than generalises.
        noise = random.getrandbits(64)
        rng = random.Random((self._seed ^ noise) * 1_000_003 ^ idx)
        word = self.words[idx]
        state = sample_state(word, rng, p_reveal=self._p_reveal(word))

        ids, attn = encode_masked(state.masked, self.max_len)
        guessed_vec = encode_guessed(state.guessed)
        target, loss_mask = encode_target(state)

        return {
            "token_ids":   torch.from_numpy(ids),
            "attn_mask":   torch.from_numpy(attn),
            "guessed_vec": torch.from_numpy(guessed_vec),
            "target":      torch.from_numpy(target),
            "loss_mask":   torch.from_numpy(loss_mask),
        }
