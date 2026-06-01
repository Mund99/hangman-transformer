"""
Vocabulary constants and character-level encoding utilities.

Token vocabulary (size 28):
  0 = PAD
  1 = '_'  (masked position)
  2..27 = 'a'..'z'
"""
from __future__ import annotations

import string

import numpy as np

PAD_ID = 0
MASK_ID = 1
LETTER_START = 2
VOCAB_SIZE = LETTER_START + 26   # 28

LETTERS = string.ascii_lowercase
LETTER_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(LETTERS)}

MAX_WORD_LEN = 45   # supports words up to 45 chars (e.g. pneumonoultramicroscopicsilicovolcanoconiosis)


def char_to_token(c: str) -> int:
    if c == "_":
        return MASK_ID
    return LETTER_START + LETTER_TO_IDX[c]


def encode_masked(masked: str, max_len: int = MAX_WORD_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Return (token_ids, attention_mask) padded to max_len."""
    ids = np.zeros(max_len, dtype=np.int64)
    attn = np.zeros(max_len, dtype=np.int64)
    n = min(len(masked), max_len)
    for i in range(n):
        ids[i] = char_to_token(masked[i])
        attn[i] = 1
    return ids, attn


def encode_guessed(guessed: set[str]) -> np.ndarray:
    """26-dim binary vector — 1 means the letter has already been guessed."""
    v = np.zeros(26, dtype=np.float32)
    for c in guessed:
        v[LETTER_TO_IDX[c]] = 1.0
    return v
