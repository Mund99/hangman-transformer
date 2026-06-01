# Result Logs

Curated logs from the experiment runs — the small, reviewable artifacts (run summaries and
per-model test evaluations). Raw training logs and checkpoints are not committed (they run to
several GB); these are the numbers behind the documentation.

## What's here

| Folder | Run | Key result |
|--------|-----|-----------|
| `sup_search5_*` | Scale-up to d=384, d=512 | sup5_b d=512 → 68.79% |
| `sup_search6_*` | Aggressive 16×/4× curriculum | 65.78% (regression) |
| `sup_search7_*` | Dedicated short-word model | regressed vs full model |
| `sup_search8_*` | Final scaling d=768, d=1024 | **sup8_a d=768 → 69.02% (best)**; d=1024 regressed |
| `sup_finetune_*` | Rare-letter reweighting | 66.32% (+0.12pp) |
| `sup_search4_*` | Step scaling at d=256 | 66.20% ceiling |
| `rl_search2,3,5,6_*` | PPO / KL-PPO / GRPO | all failed to beat supervised |

## File types

- `summary.log` — the run launcher's summary (configs + final numbers)
- `*_test_eval.log` — full test-set evaluation for a model, including the per-length
  and per-letter win-rate breakdown used on the
  [Performance Analysis](https://mund99.github.io/hangman-transformer/docs/analysis) page

For the interpreted story behind these numbers, see the
[Experiments & Findings](https://mund99.github.io/hangman-transformer/docs/experiments) page.
