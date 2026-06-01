"""
hangman — core package for the Hangman RL project.

Submodules
----------
vocab    — vocabulary constants and character encoding
env      — HangmanGame (the environment)
dataset  — state sampling, HangmanStateDataset, word loading
model    — HangmanPolicy Transformer
eval     — evaluate_win_rate, evaluate_with_wins
agent    — NeuralAgent (drop-in for the hangman API)
"""
from hangman.env import HangmanGame, make_masked, MAX_WRONG
from hangman.vocab import (
    PAD_ID, MASK_ID, VOCAB_SIZE, LETTERS, LETTER_TO_IDX,
    MAX_WORD_LEN, encode_masked, encode_guessed,
)
from hangman.dataset import (
    load_word_list, sample_state, HangmanState, HangmanStateDataset,
)
from hangman.model import HangmanPolicy, ModelConfig, count_parameters
from hangman.eval import evaluate_win_rate, evaluate_with_wins
from hangman.agent import NeuralAgent

__all__ = [
    "HangmanGame", "make_masked", "MAX_WRONG",
    "PAD_ID", "MASK_ID", "VOCAB_SIZE", "LETTERS", "LETTER_TO_IDX",
    "MAX_WORD_LEN", "encode_masked", "encode_guessed",
    "load_word_list", "sample_state", "HangmanState", "HangmanStateDataset",
    "HangmanPolicy", "ModelConfig", "count_parameters",
    "evaluate_win_rate", "evaluate_with_wins",
    "NeuralAgent",
]
