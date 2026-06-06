# Scripts

The training, evaluation, and experiment-runner scripts. Every `run_*.sh` opens with a header
documenting its hypothesis and context; this file is the map.

For the interpreted story behind these runs see the
[Experiments documentation](https://mund99.github.io/hangman-transformer/docs/experiments);
for the raw numbers see [`../results/`](../results/).

## Core scripts

| Script | What it does |
|--------|--------------|
| `train_supervised.py` | Stage 1 — supervised pretraining (multi-label BCE on sampled board states) |
| `train_rl.py` | Stage 2 — RL fine-tuning (REINFORCE / PPO / KL-PPO / GRPO) |
| `evaluate.py` | Greedy evaluation on the held-out test set (the final win-rate number) |
| `evaluate_ensemble.py` | Length-routed ensemble eval (short-word model + main model) |
| `evaluate_search.py` | Inference-time candidate-pool search variants |
| `analyze_letter_bias.py` | Per-letter win-rate breakdown (j/k/w weakness) |
| `prepare_data.py` | Build the 90/5/5 train/val/test splits from the raw corpus |
| `watch_progress.py` | Live training monitor |
| `run_training.sh` | Full pipeline orchestrator: supervised → RL → evaluation |

> The statistical baselines live separately in [`../baselines/`](../baselines/) (numpy-only).

## Experiment runs, in order

The project ran as a sequence of hypothesis-driven searches. Read top to bottom for the story.

### Supervised search at d=256 — finding the ceiling
| Script | Hypothesis | Result |
|--------|-----------|--------|
| `run_supervised_search.sh` | Curriculum + step-count ablation | 4×/2× curriculum +2.9pp |
| `run_supervised_search2.sh` | Fill the 2×2 factorial of the above | — |
| `run_supervised_search3.sh` | Map the convergence ceiling (251k corpus) | 68.12% (251k test set) |
| `run_supervised_search4.sh` | Expanded 353k corpus, max_len=45 | **66.20%** ceiling at d=256 |

### Fixing the rare-letter weakness
| Script | Hypothesis | Result |
|--------|-----------|--------|
| `run_sup_finetune.sh` | Up-weight rare letters (j/k/w/z) to fix bias | 66.32% (+0.12pp) — intrinsic, not fixable |

### RL fine-tuning — six failures
| Script | Approach | Result |
|--------|----------|--------|
| `run_rl_search1.sh` | PPO from the d=256 checkpoint | Degraded immediately |
| `run_rl_search2.sh` | Terminal reward + structural fixes | Sparse reward → gradient ≈ 0 |
| `run_rl_search3.sh` | Shaped reward + fixed config | Batch-norm advantage conflates difficulty |
| `run_rl_search4.sh` | GRPO grouped advantage | Aux-BCE degraded equilibrium |
| `run_rl_search5.sh` | KL-penalised PPO (RLHF-style) | Plateaued at baseline |
| `run_rl_search6.sh` | Stronger KL + larger val eval | Never beat supervised — **RL closed** |

### Architecture scaling — what actually worked
| Script | Hypothesis | Result |
|--------|-----------|--------|
| `run_sup_search5.sh` | Scale width: d=384, d=512 | d=384 → 68.24%, **d=512 → 68.79%** |
| `run_sup_search6.sh` | Aggressive 16×/4× curriculum for short words | 65.78% — **regression** |
| `run_sup_search7.sh` | Dedicated short-word model + ensemble | Regressed vs the full model |
| `run_sup_search8.sh` | Final scaling: d=768, d=1024 | **d=768 → 69.02% (best)**; d=1024 → 67.69% regressed |

## The one-line takeaway

Architecture scaling (to d=768) was the only reliable lever. RL, aggressive curriculum,
inference-time search, and short-word specialisation all failed or regressed — the corpus,
not the method, is the ceiling.
