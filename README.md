# Hangman Transformer

A character-level Transformer trained to play Hangman on words it has never seen.
After 20+ experiments — supervised scaling, six reinforcement-learning attempts,
curriculum tuning, inference-time search, and model specialisation — the best model
reaches a **69.02% win rate** on 17,652 held-out test words.

**[📖 Documentation & write-up](https://mund99.github.io/hangman-transformer/)** ·
**[🎮 Play in your browser](https://mund99.github.io/hangman-transformer/play)**

The in-browser demo runs the model client-side via ONNX Runtime — no server.

---

## Results

| Model | Params | Test win rate |
|-------|--------|---------------|
| Statistical baseline (CandidateFrequency) | — | 18.05% |
| Statistical baseline (NGramGlobal) | — | 39.39% |
| d=256 Transformer | 3.25M | 66.20% |
| d=512 Transformer | 19.2M | 68.79% |
| **d=768 Transformer (best)** | **43.2M** | **69.02%** |
| d=1024 Transformer | 76.8M | 67.69% (regressed) |

The headline finding: **architecture scaling was the only reliable lever.** RL, aggressive
curriculum, inference-time search, and a dedicated short-word model all failed or regressed.
The reasoning behind every run is in the [documentation](https://mund99.github.io/hangman-transformer/docs/experiments).

---

## Repository layout

```
hangman/        Core package — model, dataset, env, vocab, eval
scripts/        Training (train_supervised.py, train_rl.py) + evaluation + run_*.sh configs
baselines/      Standalone statistical baselines (numpy only)
data/           Word lists — 90/5/5 train / val / test split
results/        Curated result logs (summary + per-model test evals)
docs-site/      The documentation site (React + Vite, deployed to GitHub Pages)
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Statistical baselines (numpy only, no training needed)
python3 -u baselines/evaluate_ngram.py

# 2. Train the best model (d=768) — needs a GPU
python scripts/train_supervised.py \
  --train-file data/train_clean1.txt --val-file data/val_words.txt \
  --d-model 768 --n-layers 6 --n-heads 12 --dim-ff 3072 \
  --steps 200000 --lr 7e-5 --length-curriculum \
  --ckpt-dir logs/d768

# 3. Evaluate a checkpoint on the held-out test set
python scripts/evaluate.py --ckpt logs/d768/supervised_best.pt --words data/test_words.txt
```

The exact configuration for every experiment in the paper is in the `scripts/run_*.sh`
files (e.g. `run_sup_search8.sh` reproduces the d=768 / d=1024 scaling runs).

---

## How it works

A pre-norm Transformer encoder reads the masked word plus the set of guessed letters and
outputs a probability for each of the 26 letters; at each turn it guesses the highest-scoring
letter it hasn't tried. It's trained with multi-label binary cross-entropy on randomly sampled
mid-game board states. Full details in the
[Architecture](https://mund99.github.io/hangman-transformer/docs/architecture) and
[Training](https://mund99.github.io/hangman-transformer/docs/training) docs.

---

## Documentation

The full write-up — problem, data, baselines, architecture, training, the experiment log,
and per-length / per-letter analysis — lives at
**https://mund99.github.io/hangman-transformer/**, built from `docs-site/`.
