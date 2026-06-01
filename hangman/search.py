"""
Inference-time search strategies for HangmanPolicy.

Current greedy play (eval.py) picks argmax(model_logits) every step.
This module adds candidate-pool-based augmentation:

  At each guess step we know:
    - masked pattern  e.g.  "_ a _ _ _"
    - wrong letters   e.g.  {e, t}
    - revealed letters (implicit in masked pattern)

  We filter the training word list to CANDIDATE words that are consistent
  with this state, then compute a letter-frequency score over those candidates.
  Blending the model's global statistics with this local candidate signal
  can improve accuracy — especially on short / ambiguous words.

Strategies
----------
  greedy      : model argmax only (baseline, reproduces eval.py)
  freq        : candidate-frequency only (ignore model)
  hybrid(α)   : α * model_prob + (1-α) * cand_freq   (fixed blend)
  adaptive(k) : α = N_cand / (N_cand + k)            (auto-blend by pool size)
                small k → trust candidates sooner
                large k → trust model longer

Usage
-----
    pool  = CandidatePool.from_file("data/train_clean1.txt")
    agent = SearchAgent(model, device, pool, strategy="adaptive", k=50)
    win   = agent.play_game("anonymity")
"""
from __future__ import annotations

import string
from collections import defaultdict
from typing import Literal

import numpy as np
import torch

from hangman.env import MAX_WRONG
from hangman.vocab import MAX_WORD_LEN, encode_masked, encode_guessed


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────────────────────

class CandidatePool:
    """
    Fast vectorised filtering of a word list by game state.

    Internally stores each length-bucket as a numpy int8 array (chars as 0-25)
    so pattern-match and wrong-letter checks are batched numpy ops.
    """

    def __init__(self, words: list[str]):
        by_len: dict[int, list[str]] = defaultdict(list)
        for w in words:
            by_len[len(w)].append(w)

        # Pre-encode each length bucket as a (N, L) int8 array
        self._words:  dict[int, list[str]]      = {}
        self._arrays: dict[int, np.ndarray]     = {}   # (N, L) int8, chars as 0-25
        for n, bucket in by_len.items():
            arr = np.array(
                [[ord(c) - ord("a") for c in w] for w in bucket],
                dtype=np.int8,
            )
            self._words[n]  = bucket
            self._arrays[n] = arr

    @classmethod
    def from_file(cls, path: str) -> "CandidatePool":
        words = [w.strip() for w in open(path) if w.strip()]
        return cls(words)

    def filter_mask(self, masked: str, wrong: set[str]) -> np.ndarray:
        """
        Return a boolean mask over the length-n bucket.
        Returns (mask, n) — call with _words[n][mask] to get word strings.
        """
        n = len(masked)
        if n not in self._arrays:
            return np.zeros(0, dtype=bool), n

        chars  = self._arrays[n]          # (N, n)
        mask   = np.ones(len(self._words[n]), dtype=bool)

        # 1. Exclude words containing any wrong letter
        for c in wrong:
            ci = ord(c) - ord("a")
            mask &= ~np.any(chars == ci, axis=1)

        # 2. Enforce revealed-letter positions
        for i, mc in enumerate(masked):
            if mc != "_":
                ci = ord(mc) - ord("a")
                mask &= chars[:, i] == ci

        return mask, n

    def filter(self, masked: str, wrong: set[str]) -> list[str]:
        """Return all candidate words consistent with the current state."""
        mask, n = self.filter_mask(masked, wrong)
        if not mask.any():
            return []
        words = self._words[n]
        return [words[i] for i in np.where(mask)[0]]

    def freq_scores(self, candidates: list[str], guessed: set[str]) -> np.ndarray:
        """
        26-dim array: fraction of candidates that contain each unguessed letter.
        Vectorised: uses pre-built numpy arrays for speed.
        Guessed letters are set to 0.
        """
        scores = np.zeros(26, dtype=np.float32)
        if not candidates:
            return scores
        n = len(candidates[0])
        if n not in self._arrays:
            return scores

        # Re-derive the matching subset mask to get the numpy sub-array
        # (avoids rebuilding the char matrix from string candidates)
        # We'll just use the string-based approach for correctness; it's called
        # only for filtered candidates which are typically << full bucket size.
        guessed_set = set(guessed)
        for word in candidates:
            for c in set(word):
                if c not in guessed_set:
                    scores[ord(c) - ord("a")] += 1
        scores /= len(candidates)
        return scores

    def freq_scores_fast(
        self, masked: str, wrong: set[str], guessed: set[str]
    ) -> tuple[np.ndarray, int]:
        """
        Vectorised freq_scores — reuses the pre-built numpy arrays directly.
        Returns (scores, n_candidates).
        """
        mask, n = self.filter_mask(masked, wrong)
        n_cand = int(mask.sum())
        scores = np.zeros(26, dtype=np.float32)
        if n_cand == 0 or n not in self._arrays:
            return scores, 0

        chars = self._arrays[n][mask]       # (n_cand, word_len) int8

        # For each letter 0-25: does it appear in the word? (any across positions)
        # presence: (n_cand, 26) bool
        guessed_idx = np.array(
            [ord(c) - ord("a") for c in guessed], dtype=np.int8
        ) if guessed else np.empty(0, dtype=np.int8)

        for li in range(26):
            if li in guessed_idx:
                continue
            scores[li] = np.any(chars == li, axis=1).mean()

        return scores, n_cand


# ─────────────────────────────────────────────────────────────────────────────
# Search agent
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = Literal["greedy", "freq", "hybrid", "adaptive"]


class SearchAgent:
    """
    Plays hangman games using a trained model + optional candidate-pool search.

    Parameters
    ----------
    model    : trained HangmanPolicy (eval mode)
    device   : torch device
    pool     : CandidatePool built from the training word list
    strategy :
        "greedy"   — model argmax only (reproduces existing baseline)
        "freq"     — candidate frequency only (no model)
        "hybrid"   — fixed blend: α * model + (1-α) * freq
        "adaptive" — α = N_cand / (N_cand + k), auto-adjusts by pool size
    alpha    : fixed blend weight for "hybrid" strategy (0=pure freq, 1=pure model)
    k        : pool-size midpoint for "adaptive" (smaller k → trust candidates sooner)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        pool: CandidatePool,
        strategy: STRATEGIES = "adaptive",
        alpha: float = 0.5,
        k: float = 50.0,
        max_wrong: int = MAX_WRONG,
    ):
        self.model     = model
        self.device    = device
        self.pool      = pool
        self.strategy  = strategy
        self.alpha     = alpha
        self.k         = k
        self.max_wrong = max_wrong

    @torch.no_grad()
    def _model_probs(self, masked: str, guessed: set[str]) -> np.ndarray:
        """Return softmax probabilities over 26 letters (guessed → 0)."""
        ids, am = encode_masked(masked, MAX_WORD_LEN)
        gv      = encode_guessed(guessed)

        tok = torch.from_numpy(ids[None, :]).to(self.device)
        atm = torch.from_numpy(am[None,  :]).to(self.device)
        gvt = torch.from_numpy(gv[None,  :]).to(self.device)

        logits = self.model(tok, atm, gvt)[0]               # (26,)
        logits = logits.masked_fill(gvt[0].bool(), float("-inf"))
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs                                          # (26,) float32

    def _pick(self, masked: str, guessed: set[str], wrong: set[str]) -> str:
        """Core letter selection given current game state."""
        letters = string.ascii_lowercase

        # ── greedy: pure model ────────────────────────────────────────────────
        if self.strategy == "greedy":
            probs = self._model_probs(masked, guessed)
            return letters[int(probs.argmax())]

        # ── all other strategies need the candidate pool ──────────────────────
        # Use vectorised path: single numpy scan instead of Python loops
        freq, n_cand = self.pool.freq_scores_fast(masked, wrong, guessed)
        candidates   = None   # avoid materialising strings unless needed

        if self.strategy == "freq":
            # Pure candidate frequency; fall back to model if pool is empty
            if n_cand > 0 and freq.max() > 0:
                return letters[int(freq.argmax())]
            # Fallback: model
            probs = self._model_probs(masked, guessed)
            return letters[int(probs.argmax())]

        # ── hybrid / adaptive: blend model + frequency ────────────────────────
        probs = self._model_probs(masked, guessed)

        if self.strategy == "hybrid":
            alpha = self.alpha
        else:  # adaptive
            # alpha → 1 (pure model) when pool is large or empty
            # alpha → 0 (pure freq)  when pool is very small
            if n_cand == 0:
                alpha = 1.0
            else:
                alpha = n_cand / (n_cand + self.k)

        # Normalise freq to sum=1 over unguessed letters
        freq_sum = freq.sum()
        if freq_sum > 0:
            freq_norm = freq / freq_sum
        else:
            # freq is all-zero → fall back to pure model
            freq_norm = freq
            alpha     = 1.0

        score = alpha * probs + (1.0 - alpha) * freq_norm

        # Zero out guessed letters
        for c in guessed:
            score[ord(c) - ord("a")] = 0.0

        # Safety: if all scores are zero (shouldn't happen after fix above, but just in case)
        if score.max() <= 0:
            score = probs.copy()
            for c in guessed:
                score[ord(c) - ord("a")] = 0.0

        return letters[int(score.argmax())]

    def play_game(self, word: str) -> bool:
        """Play one game. Returns True if won."""
        revealed: set[str] = set()
        wrong:    set[str] = set()

        for _ in range(26):
            if all(c in revealed for c in word):
                return True
            if len(wrong) >= self.max_wrong:
                return False

            masked  = "".join(c if c in revealed else "_" for c in word)
            guessed = revealed | wrong
            letter  = self._pick(masked, guessed, wrong)

            if letter in word:
                revealed.add(letter)
            else:
                wrong.add(letter)

        return all(c in revealed for c in word)

    def evaluate(
        self,
        words: list[str],
        verbose: bool = False,
    ) -> dict:
        """Evaluate win rate over a word list. Returns metrics dict."""
        wins = []
        for w in words:
            won = self.play_game(w)
            wins.append(won)
            if verbose:
                status = "✓" if won else "✗"
                print(f"  {status}  {w}")

        win_rate = sum(wins) / max(1, len(wins))
        return {
            "win_rate":  win_rate,
            "n_wins":    sum(wins),
            "n_games":   len(wins),
            "wins":      wins,
        }
