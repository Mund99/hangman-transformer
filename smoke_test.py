"""
End-to-end smoke test — exercises every module in under a minute.

    python smoke_test.py
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hangman.dataset import HangmanStateDataset, load_word_list, sample_state
from hangman.model import HangmanPolicy, ModelConfig, count_parameters
from hangman.eval import evaluate_win_rate
from scripts.train_rl import collect_rollout
from utils import pick_device


def main():
    print("[1] device")
    device = pick_device("auto")
    print(f"    {device}")

    print("[2] data")
    train = load_word_list("data/train_words.txt")
    val   = load_word_list("data/val_words.txt")[:200]
    print(f"    train={len(train):,}  val_sample={len(val)}")

    print("[3] state sampler")
    rng = random.Random(0)
    for w in random.sample(train, 3):
        s = sample_state(w, rng)
        print(f"    {w:14s} -> {s.masked:14s}  guessed={sorted(s.guessed)}")

    print("[4] model forward")
    cfg   = ModelConfig(d_model=128, n_heads=4, n_layers=2, dim_ff=256)
    model = HangmanPolicy(cfg).to(device)
    print(f"    params={count_parameters(model):,}")

    ds     = HangmanStateDataset(train[:1024])
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    batch  = next(iter(loader))
    for k in batch:
        batch[k] = batch[k].to(device)
    logits = model(batch["token_ids"], batch["attn_mask"], batch["guessed_vec"])
    print(f"    logits.shape={tuple(logits.shape)}  finite={torch.isfinite(logits).all().item()}")

    print("[5] one supervised step")
    opt  = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bce  = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
    loss = (bce * batch["loss_mask"]).sum() / batch["loss_mask"].sum().clamp(min=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"    loss={loss.item():.4f}")

    print("[6] one batch of RL games (8 games)")
    t = time.time()
    result = collect_rollout(
        model, random.sample(train, 8), device,
        win_bonus=5.0, loss_penalty=-2.0,
    )
    if result is not None:
        _, _, _, _, _, flat_ret, wins = result
        print(f"    wins={sum(wins)}/{len(wins)}  total_steps={flat_ret.numel()}  took={time.time()-t:.2f}s")
    else:
        print(f"    (empty rollout)  took={time.time()-t:.2f}s")

    print("[7] offline eval on val sample")
    t = time.time()
    metrics = evaluate_win_rate(model, val, device)
    print(f"    (untrained tiny model) val_win_rate={metrics['win_rate']*100:.2f}%  took={time.time()-t:.1f}s")

    print("\n[ok] smoke test passed.")


if __name__ == "__main__":
    main()
