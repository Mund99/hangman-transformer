"""
Drop-in inference agent for the HangmanAPI notebook.

Usage:
    from hangman.agent import NeuralAgent
    agent = NeuralAgent("checkpoints/rl_best.pt")
    agent.bind_env(api)
    api.start_game(agent=agent, practice=1, verbose=True)
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import torch

from hangman.vocab import MAX_WORD_LEN, encode_masked, encode_guessed
from hangman.model import HangmanPolicy, ModelConfig


def _device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NeuralAgent:
    """
    Wraps a trained HangmanPolicy for use with the hangman API.

    The API class tracks guessed_letters externally; we read it at guess time
    via self.env.guessed_letters, which mirrors the notebook's agent contract.
    """

    def __init__(self, ckpt_path: str | Path, device: str | None = None, env=None):
        self.device = _device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ModelConfig(**ckpt["cfg"])
        self.model = HangmanPolicy(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.env = env
        self._letters = string.ascii_lowercase

    def bind_env(self, env) -> None:
        self.env = env

    @torch.no_grad()
    def guess(self, state: str) -> str:
        """
        state : current word with underscores, e.g. "_ a p _ l e" or "_ap_le".
        Returns the highest-scoring unguessed letter.
        """
        masked  = "".join(c for c in state if c.isalpha() or c == "_")
        guessed = getattr(self.env, "guessed_letters", set()) if self.env is not None else set()

        ids, am = encode_masked(masked, MAX_WORD_LEN)
        gv      = encode_guessed(set(guessed))

        tok    = torch.from_numpy(ids[None, :]).to(self.device)
        atm    = torch.from_numpy(am[None, :]).to(self.device)
        gvt    = torch.from_numpy(gv[None, :]).to(self.device)

        logits = self.model(tok, atm, gvt)[0]
        logits = logits.masked_fill(gvt[0].bool(), float("-inf"))
        return self._letters[int(logits.argmax().item())]
