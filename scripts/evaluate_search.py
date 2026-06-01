"""
Inference-time search evaluation — compares multiple strategies on the test set.

Context
-------
Best model so far: sup5_b (d=512, 19.2M params) — test win rate 68.79%.
All gains so far came from better training (architecture, curriculum, data).
This script tests whether inference-time candidate-pool search can push higher
WITHOUT any retraining.

Strategies tested
-----------------
  greedy       — model argmax only (reproduces current 68.79% baseline)
  freq         — candidate word-list frequency only (no model)
  adaptive-10  — α = N/(N+10):  trusts candidates when pool ≤ ~10 words
  adaptive-50  — α = N/(N+50):  trusts candidates when pool ≤ ~50 words  ← default
  adaptive-200 — α = N/(N+200): trusts candidates more conservatively
  hybrid-0.3   — fixed 30% model / 70% candidate freq
  hybrid-0.7   — fixed 70% model / 30% candidate freq

Candidate pool
--------------
  data/train_clean1.txt (317k words).
  Test words are disjoint from training — the pool never contains the exact
  answer, but DOES contain same-length words matching the revealed pattern.
  That letter-frequency signal is the key inference-time boost.

Usage
-----
    # Full test set, all strategies:
    python scripts/evaluate_search.py \
        --ckpt logs/sup_search5_20260521_0116/sup5_b_d512_L6/checkpoints/supervised_best.pt

    # Quick smoke-test on 500 words:
    python scripts/evaluate_search.py \
        --ckpt logs/.../supervised_best.pt --n 500

    # Single strategy:
    python scripts/evaluate_search.py \
        --ckpt logs/.../supervised_best.pt --strategy adaptive --k 50
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from hangman.dataset import load_word_list
from hangman.model import HangmanPolicy, ModelConfig
from hangman.search import CandidatePool, SearchAgent
from utils import pick_device


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def stratify_by_length(words, wins):
    by_len = defaultdict(lambda: [0, 0])
    for w, won in zip(words, wins):
        by_len[len(w)][0] += 1
        by_len[len(w)][1] += int(won)
    return {L: (n, k, k / max(1, n)) for L, (n, k) in sorted(by_len.items())}


def bar(r, width=20):
    filled = int(r * width)
    return "█" * filled + "░" * (width - filled)


def _report_strategy(name, words, result, baseline_wr, log_lines):
    wins     = result["wins"]
    wr       = result["win_rate"]
    n_wins   = result["n_wins"]
    n_games  = result["n_games"]
    delta    = wr - baseline_wr
    sign     = "+" if delta >= 0 else ""
    verdict  = "▲ IMPROVEMENT" if delta > 0.001 else ("▼ REGRESSION" if delta < -0.001 else "≈ same")

    by_len = stratify_by_length(words, wins)

    W = 62
    lines = [
        "",
        f"  ── {name} ──",
        f"  WIN RATE : {wr*100:.2f}%  ({n_wins:,}/{n_games:,})  Δ={sign}{delta*100:+.2f}pp  {verdict}",
        f"  {'LEN':>4}  {'N':>6}  {'WINS':>5}  {'WIN%':>7}  {'DELTA':>8}  BAR",
        f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*20}",
    ]
    for L, (n, k, r) in by_len.items():
        d = r - baseline_wr
        sign2 = "+" if d >= 0 else ""
        flag = "  ◄ weak" if r < 0.40 else ""
        lines.append(f"  {L:>4}  {n:>6,}  {k:>5,}  {r*100:>6.1f}%  {sign2}{d*100:>+6.1f}pp  {bar(r)}{flag}")

    for l in lines:
        print(l)
    log_lines.extend(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate inference-time search strategies")
    ap.add_argument("--ckpt",     required=True,  help="Path to model checkpoint")
    ap.add_argument("--words",    default="data/test_words.txt")
    ap.add_argument("--pool",     default="data/train_clean1.txt",
                    help="Candidate word list (default: train_clean1.txt)")
    ap.add_argument("--n",        type=int, default=0,
                    help="Limit to first N words (0=all)")
    ap.add_argument("--strategy", default="all",
                    choices=["all","greedy","freq","adaptive","hybrid"],
                    help="Which strategy to run (default: all)")
    ap.add_argument("--k",        type=float, default=50.0,
                    help="Adaptive pool-size midpoint (only used with --strategy adaptive)")
    ap.add_argument("--alpha",    type=float, default=0.5,
                    help="Hybrid blend alpha (only used with --strategy hybrid)")
    ap.add_argument("--device",   default="auto", choices=["auto","mps","cuda","cpu"])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[device]  {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg  = ModelConfig(**ckpt["cfg"])
    model = HangmanPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[model]   {args.ckpt}  step={ckpt.get('step','?')}  val={ckpt.get('val_win_rate',0)*100:.2f}%")

    # ── Load word lists ───────────────────────────────────────────────────────
    words = load_word_list(args.words)
    if args.n > 0:
        words = words[:args.n]
    print(f"[words]   {len(words):,} test words")

    print(f"[pool]    Loading candidate pool from {args.pool} …", end=" ", flush=True)
    t0   = time.time()
    pool = CandidatePool.from_file(args.pool)
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Strategy table ────────────────────────────────────────────────────────
    if args.strategy == "all":
        strategies = [
            ("greedy",       dict(strategy="greedy")),
            ("freq",         dict(strategy="freq")),
            ("adaptive-k10", dict(strategy="adaptive", k=10)),
            ("adaptive-k50", dict(strategy="adaptive", k=50)),
            ("adaptive-k200",dict(strategy="adaptive", k=200)),
            ("hybrid-a0.3",  dict(strategy="hybrid",   alpha=0.3)),
            ("hybrid-a0.7",  dict(strategy="hybrid",   alpha=0.7)),
        ]
    elif args.strategy == "adaptive":
        strategies = [(f"adaptive-k{args.k:.0f}", dict(strategy="adaptive", k=args.k))]
    elif args.strategy == "hybrid":
        strategies = [(f"hybrid-a{args.alpha}", dict(strategy="hybrid", alpha=args.alpha))]
    else:
        strategies = [(args.strategy, dict(strategy=args.strategy))]

    # ── Run all strategies ────────────────────────────────────────────────────
    W = 62
    log_lines = []
    header = [
        "",
        "=" * W,
        "  INFERENCE-TIME SEARCH EVALUATION",
        f"  Checkpoint : {args.ckpt}",
        f"  Test words : {args.words}  ({len(words):,} words)",
        f"  Cand pool  : {args.pool}",
        "=" * W,
    ]
    for l in header:
        print(l)
    log_lines.extend(header)

    results = {}
    baseline_wr = 0.0

    for name, kwargs in strategies:
        agent = SearchAgent(model, device, pool, **kwargs)
        print(f"\n  Running '{name}' …", end=" ", flush=True)
        t0     = time.time()
        result = agent.evaluate(words)
        elapsed = time.time() - t0
        print(f"{result['win_rate']*100:.2f}%  ({elapsed:.0f}s)")
        results[name] = result

        # First strategy (greedy) sets the baseline delta
        if name == "greedy":
            baseline_wr = result["win_rate"]

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = [
        "",
        "=" * W,
        "  STRATEGY COMPARISON SUMMARY",
        f"  {'STRATEGY':<18}  {'WIN%':>7}  {'DELTA':>9}  {'BAR':<20}",
        f"  {'-'*18}  {'-'*7}  {'-'*9}  {'-'*20}",
    ]
    for name, result in results.items():
        wr    = result["win_rate"]
        delta = wr - baseline_wr
        sign  = "+" if delta >= 0 else ""
        verdict = " ▲" if delta > 0.001 else (" ▼" if delta < -0.001 else "")
        summary.append(
            f"  {name:<18}  {wr*100:>6.2f}%  {sign}{delta*100:>+7.2f}pp  {bar(wr)}{verdict}"
        )
    summary += ["", "=" * W, ""]
    for l in summary:
        print(l)
    log_lines.extend(summary)

    # ── Per-length breakdown for each strategy ─────────────────────────────────
    print("\n  ── PER-LENGTH BREAKDOWN ──\n")
    log_lines.append("\n  ── PER-LENGTH BREAKDOWN ──\n")
    for name, result in results.items():
        _report_strategy(name, words, result, baseline_wr, log_lines)

    # ── Save log ───────────────────────────────────────────────────────────────
    log_path = args.log_file
    if log_path is None:
        ckpt_dir  = Path(args.ckpt).parent.parent   # .../exp_name/checkpoints/...
        log_dir   = ckpt_dir / "search_eval"
        log_dir.mkdir(parents=True, exist_ok=True)
        n_tag     = f"_n{args.n}" if args.n > 0 else ""
        log_path  = str(log_dir / f"search_eval{n_tag}.log")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  [saved]  {log_path}")


if __name__ == "__main__":
    main()
