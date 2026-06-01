"""
Shared evaluation utilities used by train_supervised, train_rl, and evaluate.
"""
from __future__ import annotations

import numpy as np
import torch

from hangman.vocab import MAX_WORD_LEN, encode_masked, encode_guessed
from hangman.env import MAX_WRONG


@torch.no_grad()
def evaluate_win_rate(
    model,
    words: list[str],
    device: torch.device,
    max_wrong: int = MAX_WRONG,
    batch_size: int = 512,
) -> dict[str, float]:
    """Play one greedy game per word in parallel batches. Returns aggregate metrics."""
    model.eval()
    wins = 0
    total_wrong = 0
    total_correct = 0
    n = len(words)

    for start in range(0, n, batch_size):
        batch = words[start : start + batch_size]
        revealed = [set() for _ in batch]
        wrong    = [set() for _ in batch]
        done     = [False] * len(batch)
        result   = [False] * len(batch)

        for _step in range(26):
            active = [i for i, d in enumerate(done) if not d]
            if not active:
                break

            tok_list, am_list, gv_list = [], [], []
            for i in active:
                w      = batch[i]
                masked = "".join(c if c in revealed[i] else "_" for c in w)
                ids, am = encode_masked(masked, MAX_WORD_LEN)
                gv      = encode_guessed(revealed[i] | wrong[i])
                tok_list.append(ids)
                am_list.append(am)
                gv_list.append(gv)

            tok = torch.from_numpy(np.stack(tok_list)).to(device)
            am  = torch.from_numpy(np.stack(am_list)).to(device)
            gv  = torch.from_numpy(np.stack(gv_list)).to(device)

            logits  = model(tok, am, gv)
            logits  = logits.masked_fill(gv.bool(), float("-inf"))
            actions = logits.argmax(dim=-1).cpu().tolist()

            for slot, i in enumerate(active):
                letter = chr(ord("a") + actions[slot])
                w = batch[i]
                if letter in w:
                    revealed[i].add(letter)
                    total_correct += 1
                    if all(c in revealed[i] for c in w):
                        done[i] = True
                        result[i] = True
                        wins += 1
                else:
                    wrong[i].add(letter)
                    total_wrong += 1
                    if len(wrong[i]) >= max_wrong:
                        done[i] = True
                        result[i] = False

        for i, d in enumerate(done):
            if not d:
                result[i] = all(c in revealed[i] for c in batch[i])
                if result[i]:
                    wins += 1

    model.train()
    return {
        "win_rate":            wins / max(1, n),
        "n_games":             n,
        "avg_wrong_per_game":  total_wrong  / max(1, n),
        "avg_correct_per_game": total_correct / max(1, n),
    }


@torch.no_grad()
def evaluate_with_wins(
    model,
    words: list[str],
    device: torch.device,
    max_wrong: int = MAX_WRONG,
    batch_size: int = 512,
) -> list[bool]:
    """Same as evaluate_win_rate but returns a per-word win flag list."""
    model.eval()
    all_wins = [False] * len(words)

    for start in range(0, len(words), batch_size):
        batch    = words[start : start + batch_size]
        revealed = [set() for _ in batch]
        wrong    = [set() for _ in batch]
        done     = [False] * len(batch)
        result   = [False] * len(batch)

        for _step in range(26):
            active = [i for i, d in enumerate(done) if not d]
            if not active:
                break

            tok_list, am_list, gv_list = [], [], []
            for i in active:
                masked = "".join(c if c in revealed[i] else "_" for c in batch[i])
                ids, am = encode_masked(masked, MAX_WORD_LEN)
                gv      = encode_guessed(revealed[i] | wrong[i])
                tok_list.append(ids)
                am_list.append(am)
                gv_list.append(gv)

            tok = torch.from_numpy(np.stack(tok_list)).to(device)
            am  = torch.from_numpy(np.stack(am_list)).to(device)
            gv  = torch.from_numpy(np.stack(gv_list)).to(device)

            logits  = model(tok, am, gv)
            logits  = logits.masked_fill(gv.bool(), float("-inf"))
            actions = logits.argmax(dim=-1).cpu().tolist()

            for slot, i in enumerate(active):
                letter = chr(ord("a") + actions[slot])
                w = batch[i]
                if letter in w:
                    revealed[i].add(letter)
                    if all(c in revealed[i] for c in w):
                        done[i] = True
                        result[i] = True
                else:
                    wrong[i].add(letter)
                    if len(wrong[i]) >= max_wrong:
                        done[i] = True
                        result[i] = False

        for j, i in enumerate(range(start, start + len(batch))):
            all_wins[i] = result[j]

    return all_wins
