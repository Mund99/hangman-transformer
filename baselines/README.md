# Statistical Baselines

Non-neural reference points for Hangman. These are **not** the model — they're the
rule-based strategies the Transformer is measured against, so the 69% result has
context.

## The two agents

| Agent | Idea | Win rate |
|-------|------|----------|
| **CandidateFrequency** | Filter the training dictionary to words matching the current pattern, then guess the most common letter among them. | **18.05%** |
| **NGramGlobal** | Build weighted positional n-gram tables (bigram–9-gram) from all training words; each window votes for letters that fill its blanks. | **39.39%** |
| Transformer (best) | — | **69.02%** |

All three are evaluated on the same 17,652-word held-out test set, which is fully
disjoint from the training corpus.

## Why they fall short

- **CandidateFrequency** collapses on short words: a 4-letter word matches few
  dictionary entries, and a single wrong guess can wipe out the entire candidate set.
- **NGramGlobal** is stronger because positional n-grams capture spelling structure,
  but it has no notion of the *whole word* — it scores each window independently.

The neural model wins because it conditions on the entire board at once and learns
orthographic patterns that generalise to words it has never seen.

## Run it

From the repo root (needs `numpy`):

```bash
python3 -u baselines/evaluate_ngram.py            # both agents
python3 -u baselines/evaluate_ngram.py --agent freq
python3 -u baselines/evaluate_ngram.py --agent ngram
```

Uses `data/train_clean1.txt` (317k words) and `data/test_words.txt` (17,652 words),
both included in this repo. The full evaluation takes a few minutes per agent.
