"""
Hangman game environment.

MAX_WRONG = 6 — the standard limit. Classic hangman draws 6 body
parts (head, body, 2 arms, 2 legs) before the figure is complete.
Some variants add a gallows for 7, but this project uses 6.
"""
from __future__ import annotations

MAX_WRONG = 6


def make_masked(word: str, revealed: set[str]) -> str:
    return "".join(c if c in revealed else "_" for c in word)


class HangmanGame:
    """Stateful single game of hangman."""

    def __init__(self, word: str, max_wrong: int = MAX_WRONG):
        self.word = word
        self.max_wrong = max_wrong
        self.revealed: set[str] = set()
        self.wrong: set[str] = set()

    @property
    def guessed(self) -> set[str]:
        return self.revealed | self.wrong

    @property
    def masked(self) -> str:
        return make_masked(self.word, self.revealed)

    @property
    def won(self) -> bool:
        return all(c in self.revealed for c in self.word)

    @property
    def lost(self) -> bool:
        return len(self.wrong) >= self.max_wrong

    @property
    def done(self) -> bool:
        return self.won or self.lost

    @property
    def lives_remaining(self) -> int:
        return self.max_wrong - len(self.wrong)

    def guess(self, letter: str) -> bool:
        """Apply a guess. Returns True iff the letter is in the word."""
        if letter in self.guessed:
            return False
        if letter in self.word:
            self.revealed.add(letter)
            return True
        self.wrong.add(letter)
        return False
