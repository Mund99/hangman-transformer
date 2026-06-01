"""
Split words into train/val/test word lists.

By default, merges NLTK corpus with any extra word files (--extra).
The competition word list raw/words_250000_train.txt is included automatically
if it exists. Use --input to replace NLTK with a fully custom word file.

Usage (run from project root):
    python scripts/prepare_data.py                          # NLTK + raw/words_250000_train.txt
    python scripts/prepare_data.py --extra raw/medical.txt  # add another source
    python scripts/prepare_data.py --input raw/custom.txt   # replace NLTK entirely
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

VALID_RE = re.compile(r"^[a-z]+$")
VOWELS   = set("aeiou")

# Extra word files merged in by default if present
DEFAULT_EXTRA = ["raw/words_250000_train.txt"]


def load_words(path: Path) -> set[str]:
    raw = path.read_text().split()
    return {w.strip().lower() for w in raw if w.strip()}


def has_vowel(word: str) -> bool:
    return any(c in VOWELS for c in word)


def clean(words: set[str], min_len: int, max_len: int) -> list[str]:
    out = []
    for w in words:
        if not VALID_RE.match(w):
            continue
        if not (min_len <= len(w) <= max_len):
            continue
        # Drop vowel-free words — they are unguessable in hangman
        # (all 6 wrong guesses consumed before reaching any letter)
        if not has_vowel(w):
            continue
        out.append(w)
    out.sort()
    return out


def load_nltk_words() -> set[str]:
    """Load words from NLTK corpus (words list + WordNet lemmas)."""
    try:
        import nltk
        from nltk.corpus import wordnet, words as nltk_words

        try:
            wordnet.ensure_loaded()
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk_words.words()
        except LookupError:
            nltk.download("words", quiet=True)

        words: set[str] = set()
        for w in nltk_words.words():
            w_clean = w.strip().lower()
            if VALID_RE.match(w_clean):
                words.add(w_clean)
        for syn in wordnet.all_synsets():
            for lemma in syn.lemma_names():
                if "_" not in lemma:
                    w_clean = lemma.lower()
                    if VALID_RE.match(w_clean):
                        words.add(w_clean)
        print(f"[nltk] {len(words):,} words from NLTK corpus + WordNet", file=sys.stderr)
        return words
    except Exception as e:
        sys.exit(f"NLTK required but not available: {e}\nRun: pip install nltk")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="Replace NLTK with a custom word file")
    ap.add_argument("--extra", nargs="*", default=None,
                    help="Extra word files to merge (default: raw/words_250000_train.txt if present)")
    ap.add_argument("--no-extra", action="store_true",
                    help="Skip default extra word files")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=45)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-nltk-to", default=None,
                    help="Save NLTK words to this file (e.g., raw/nltk_words.txt)")
    args = ap.parse_args()

    # ── Primary source ────────────────────────────────────────────────────────
    if args.input:
        src = Path(args.input)
        if not src.exists():
            sys.exit(f"input file not found: {src}")
        words = load_words(src)
        print(f"[load] {len(words):,} raw words from {src}")
    else:
        words = load_nltk_words()
        if args.save_nltk_to:
            out_path = Path(args.save_nltk_to)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(sorted(words)) + "\n")
            print(f"[write] NLTK words saved to {out_path}")

    # ── Extra sources ─────────────────────────────────────────────────────────
    extra_paths = [] if args.no_extra else (args.extra if args.extra is not None else DEFAULT_EXTRA)
    for ep in extra_paths:
        p = Path(ep)
        if not p.exists():
            print(f"[extra] {p} not found, skipping", file=sys.stderr)
            continue
        extra = load_words(p)
        before = len(words)
        words |= extra
        print(f"[extra] {p}: {len(extra):,} words, +{len(words)-before:,} new → {len(words):,} total")

    cleaned = clean(words, args.min_len, args.max_len)
    print(f"[clean] {len(cleaned):,} words in length [{args.min_len},{args.max_len}]")

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)

    n = len(cleaned)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test = cleaned[:n_test]
    val = cleaned[n_test : n_test + n_val]
    train = cleaned[n_test + n_val :]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in [("train", train), ("val", val), ("test", test)]:
        p = out_dir / f"{name}_words.txt"
        p.write_text("\n".join(sorted(lst)) + "\n")
        print(f"[write] {p}: {len(lst):,} words")

    print("\n[ok] splits written. Word-level overlap between splits: 0 by construction.")


if __name__ == "__main__":
    main()
