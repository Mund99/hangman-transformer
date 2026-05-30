#!/usr/bin/env python3
"""
baselines/evaluate_ngram.py

Statistical (non-neural) baselines for Hangman, evaluated on the held-out test set.
These are the reference points the Transformer is measured against — NOT the model itself.

Two agents:

  1. CandidateFrequencyAgent — filters training words matching the current masked
     pattern (and excluding wrong letters); scores candidate letters by frequency
     in that surviving set. Falls back to global letter frequency when no candidate
     words remain. This is the classic "filter the dictionary" Hangman strategy.

  2. NGramGlobalAgent — weighted positional n-gram scoring (bigram … 9-gram) built
     from the full training corpus, with no candidate filtering. Each n-gram window
     votes for the letters that could fill its blanks, weighted by corpus frequency
     and window length (n**3).

Results on the 17,652-word held-out test set (disjoint from training):

    CandidateFrequency   18.05%   avg 5.58 wrong/game
    NGramGlobal          39.39%   avg 4.99 wrong/game
    Transformer (best)   69.02%   — see the model, not this file

Usage (run from the repo root):
    python3 -u baselines/evaluate_ngram.py
    python3 -u baselines/evaluate_ngram.py --agent freq
    python3 -u baselines/evaluate_ngram.py --agent ngram
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, defaultdict

import numpy as np

MAX_WRONG = 6


def setup_logging(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Word index: O(k) candidate lookup via frozenset intersection
# ---------------------------------------------------------------------------

class WordIndex:
    def __init__(self, words: list[str], logger) -> None:
        self.words_by_length: dict[int, list[str]] = defaultdict(list)
        for w in words:
            self.words_by_length[len(w)].append(w)

        self._idx: dict[int, dict[int, dict[str, frozenset]]] = {}
        for length, bucket in self.words_by_length.items():
            idx: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
            for word_i, word in enumerate(bucket):
                for pos, char in enumerate(word):
                    idx[pos][char].add(word_i)
            self._idx[length] = {
                pos: {char: frozenset(s) for char, s in char_map.items()}
                for pos, char_map in idx.items()
            }

        logger.info(f"Word index built across {len(self.words_by_length)} length buckets")

    def candidates(self, masked: str, wrong: set[str]) -> list[str]:
        length = len(masked)
        bucket = self.words_by_length.get(length)
        if not bucket:
            return []

        idx = self._idx[length]
        live: frozenset = frozenset(range(len(bucket)))

        for pos, char in enumerate(masked):
            if char != "_":
                live = live & idx[pos].get(char, frozenset())
                if not live:
                    return []

        for letter in wrong:
            bad: set = set()
            for pos in range(length):
                bad |= idx[pos].get(letter, frozenset())
            live = live - bad
            if not live:
                return []

        return [bucket[i] for i in live]


# ---------------------------------------------------------------------------
# N-gram tables with position-indexed scoring (no regex)
# ---------------------------------------------------------------------------

class NGramTables:
    def __init__(self, words: list[str], logger, max_n: int = 9) -> None:
        self.max_n = max_n
        self.gram_chars:  dict[int, np.ndarray] = {}   # (G, n) uint8, a=0…z=25
        self.gram_counts: dict[int, np.ndarray] = {}   # (G,) int32, corpus freq

        for n in range(2, max_n + 1):
            c: Counter = Counter()
            for word in words:
                for i in range(len(word) - n + 1):
                    c[word[i : i + n]] += 1

            grams = list(c.items())
            G = len(grams)
            chars  = np.empty((G, n), dtype=np.uint8)
            counts = np.empty(G,      dtype=np.int32)
            for gi, (gram, cnt) in enumerate(grams):
                for p, ch in enumerate(gram):
                    chars[gi, p] = ord(ch) - 97
                counts[gi] = cnt

            self.gram_chars[n]  = chars
            self.gram_counts[n] = counts
            logger.info(f"  {n}-gram table built  ({G:,} unique grams)")

    def score_letters(self, masked: str, guessed: set[str]) -> Counter:
        acc = np.zeros(26, dtype=np.float64)
        n_total = len(masked)

        for n in range(min(self.max_n, n_total), 1, -1):
            weight  = n ** 3
            chars   = self.gram_chars[n]
            counts  = self.gram_counts[n]

            for i in range(n_total - n + 1):
                window    = masked[i : i + n]
                blank_pos = [j for j, ch in enumerate(window) if ch == "_"]
                known     = [(j, ord(ch) - 97) for j, ch in enumerate(window) if ch != "_"]

                if not blank_pos:
                    continue

                if known:
                    pos0, ci0 = known[0]
                    live = np.where(chars[:, pos0] == ci0)[0]
                    for pos, ci in known[1:]:
                        live = live[chars[live, pos] == ci]
                        if live.size == 0:
                            break
                else:
                    live = np.arange(len(counts))

                if live.size == 0:
                    continue

                w = counts[live].astype(np.float64) * weight
                for j in blank_pos:
                    acc += np.bincount(chars[live, j].astype(np.intp),
                                       weights=w, minlength=26)

        result: Counter = Counter()
        for li in range(26):
            if acc[li] > 0:
                letter = chr(li + 97)
                if letter not in guessed:
                    result[letter] = int(acc[li])
        return result


def global_letter_freq(words: list[str]) -> list[str]:
    c = Counter(ch for w in words for ch in set(w))
    return [letter for letter, _ in c.most_common()]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class CandidateFrequencyAgent:
    name = "CandidateFrequency"

    def __init__(self, index: WordIndex, fallback: list[str]) -> None:
        self.index = index
        self.fallback = fallback

    def guess(self, masked: str, guessed: set[str], wrong: set[str]) -> str:
        candidates = self.index.candidates(masked, wrong)
        if candidates:
            scores: Counter = Counter()
            for word in candidates:
                for pos, char in enumerate(word):
                    if masked[pos] == "_" and char not in guessed:
                        scores[char] += 1
            if scores:
                return scores.most_common(1)[0][0]

        for letter in self.fallback:
            if letter not in guessed:
                return letter
        raise RuntimeError("No letter left")


class NGramGlobalAgent:
    name = "NGramGlobal"

    def __init__(self, ngrams: NGramTables, fallback: list[str]) -> None:
        self.ngrams = ngrams
        self.fallback = fallback

    def guess(self, masked: str, guessed: set[str], wrong: set[str]) -> str:
        scores = self.ngrams.score_letters(masked, guessed)
        if scores:
            return max(scores, key=scores.get)
        for letter in self.fallback:
            if letter not in guessed:
                return letter
        raise RuntimeError("No letter left")


# ---------------------------------------------------------------------------
# Game simulation + evaluation
# ---------------------------------------------------------------------------

def play_game(agent, secret: str) -> dict:
    guessed: set[str] = set()
    revealed: set[str] = set()
    wrong:    set[str] = set()

    for _ in range(26):
        masked = "".join(c if c in revealed else "_" for c in secret)
        if "_" not in masked:
            break
        letter = agent.guess(masked, guessed, wrong)
        guessed.add(letter)
        if letter in secret:
            revealed.add(letter)
        else:
            wrong.add(letter)
            if len(wrong) >= MAX_WRONG:
                break

    won = all(c in revealed for c in secret)
    return {"won": won, "wrong": len(wrong), "correct": len(revealed)}


_LOG_INTERVAL = 500


def evaluate(agent, test_words: list[str], logger) -> dict:
    wins = total_wrong = total_correct = 0
    by_length: dict[int, dict] = defaultdict(lambda: {"wins": 0, "games": 0})
    n = len(test_words)

    t0 = time.time()
    for i, word in enumerate(test_words):
        r = play_game(agent, word)
        if r["won"]:
            wins += 1
        total_wrong   += r["wrong"]
        total_correct += r["correct"]
        b = by_length[len(word)]
        b["games"] += 1
        if r["won"]:
            b["wins"] += 1

        if (i + 1) % _LOG_INTERVAL == 0:
            done    = i + 1
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta_s   = (n - done) / rate if rate > 0 else 0
            logger.info(
                f"  {done:>6}/{n}  win={wins/done*100:5.1f}%"
                f"  {rate:5.1f} games/s  ETA={int(eta_s//60)}m{int(eta_s%60):02d}s"
            )

    return {
        "win_rate":    wins / n,
        "n_games":     n,
        "avg_wrong":   total_wrong   / n,
        "avg_correct": total_correct / n,
        "elapsed_s":   time.time() - t0,
        "by_length":   by_length,
    }


def print_report(name: str, m: dict, logger) -> None:
    logger.info("=" * 60)
    logger.info(f"Agent : {name}")
    logger.info(f"Games : {m['n_games']:,}")
    logger.info(f"Win % : {m['win_rate']*100:.2f}%")
    logger.info(f"Avg wrong/game   : {m['avg_wrong']:.3f}")
    logger.info(f"Avg correct/game : {m['avg_correct']:.3f}")
    logger.info(f"Wall time        : {m['elapsed_s']:.1f}s")
    logger.info(f"{'Length':>7}  {'Games':>6}  {'Wins':>6}  {'Win%':>6}")
    logger.info("-" * 35)
    for length in sorted(m["by_length"]):
        g = m["by_length"][length]["games"]
        w = m["by_length"][length]["wins"]
        logger.info(f"{length:>7}  {g:>6}  {w:>6}  {w/g*100:>5.1f}%")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    default="data/train_clean1.txt")
    parser.add_argument("--test",     default="data/test_words.txt")
    parser.add_argument("--agent",    choices=["freq", "ngram", "both"], default="both")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file)

    logger.info(f"Loading training words from {args.train}")
    with open(args.train) as f:
        train_words = [l.strip() for l in f if l.strip()]
    logger.info(f"  {len(train_words):,} training words")

    logger.info(f"Loading test words from {args.test}")
    with open(args.test) as f:
        test_words = [l.strip() for l in f if l.strip()]
    logger.info(f"  {len(test_words):,} test words")

    logger.info("Building word index…")
    index = WordIndex(train_words, logger)
    fallback = global_letter_freq(train_words)

    ngrams = None
    if args.agent in ("ngram", "both"):
        logger.info("Building n-gram tables…")
        ngrams = NGramTables(train_words, logger)

    results = []
    if args.agent in ("freq", "both"):
        logger.info(f"Evaluating {CandidateFrequencyAgent.name}…")
        m = evaluate(CandidateFrequencyAgent(index, fallback), test_words, logger)
        print_report(CandidateFrequencyAgent.name, m, logger)
        results.append((CandidateFrequencyAgent.name, m))

    if args.agent in ("ngram", "both") and ngrams is not None:
        logger.info(f"Evaluating {NGramGlobalAgent.name}…")
        m = evaluate(NGramGlobalAgent(ngrams, fallback), test_words, logger)
        print_report(NGramGlobalAgent.name, m, logger)
        results.append((NGramGlobalAgent.name, m))

    if results:
        logger.info("=" * 60)
        logger.info(f"{'Method':<24} {'Win%':>7}  {'Avg wrong':>10}")
        logger.info("-" * 45)
        for name, m in results:
            logger.info(f"{name:<24} {m['win_rate']*100:>6.2f}%  {m['avg_wrong']:>10.3f}")
        logger.info(f"{'Transformer (best)':<24} {69.02:>6.2f}%  {'—':>10}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
