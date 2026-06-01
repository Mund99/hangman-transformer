"""
Stage 2: PPO fine-tuning.

Initialises from the supervised checkpoint and plays full hangman games
end-to-end.  Each training step:

  1. Collect a batch of games with the current policy (no-grad rollout),
     storing states, actions, and old log-probs for PPO replay.
  2. Compute return-to-go (γ=1) and normalise advantages using a per-batch
     mean/std baseline — more responsive than EMA under rapid policy change.
  3. Run --ppo-epochs optimisation epochs over the collected batch using
     the PPO clipped surrogate objective (ε = --clip-eps, default 0.2).
     The clip prevents any single update from making a destructive policy
     change, eliminating the degradation seen with REINFORCE.
  4. Add either:
     a) Auxiliary supervised BCE loss (--aux-coef, default 0.1) on random
        hangman states to prevent catastrophic forgetting (searches 1–4), OR
     b) KL-divergence penalty (--kl-coef > 0) at the actual rollout states,
        measured against a frozen reference model (defaults to --init).
        KL(π_θ || π_ref) keeps the policy close to the supervised checkpoint
        while allowing RL to improve win rate — RLHF-style regularisation.
        Replaces aux BCE entirely; set --aux-coef 0 when using --kl-coef.

Word sampling uses a curriculum (--curriculum flag):
  35% short words (len ≤ 6)  |  35% medium (7–10)  |  30% long (≥ 11)
This forces the policy to improve on short words rather than coasting on
long-word wins.

PPO objective (per step t in each collected episode):
  ratio    = π_θ(a_t|s_t) / π_old(a_t|s_t)
  L_clip   = min(ratio × A_t,  clip(ratio, 1-ε, 1+ε) × A_t)
  L_ent    = H(π_θ(·|s_t))
  L_kl     = KL(π_θ(·|s_t) || π_ref(·|s_t))  [if --kl-coef > 0]
  loss     = -mean(L_clip) - entropy_coef × L_ent + kl_coef × L_kl

Reward shaping:
  +1.0  correct guess
  -1.0  wrong guess
  +5.0  win bonus  (added to the last step's reward)
  -2.0  loss penalty

Usage (run from project root):
    python scripts/train_rl.py --init checkpoints/supervised_best.pt --curriculum
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hangman.env import HangmanGame
from hangman.vocab import MAX_WORD_LEN, encode_masked, encode_guessed
from hangman.dataset import load_word_list, sample_state, encode_target
from hangman.model import HangmanPolicy, ModelConfig
from hangman.eval import evaluate_win_rate
from utils import pick_device, setup_logging, demo_game


def encode_batch_states(games: list[HangmanGame], device: torch.device):
    tok, am, gv = [], [], []
    for g in games:
        ids, m = encode_masked(g.masked, MAX_WORD_LEN)
        v = encode_guessed(g.guessed)
        tok.append(ids)
        am.append(m)
        gv.append(v)
    return (
        torch.from_numpy(np.stack(tok)).to(device),
        torch.from_numpy(np.stack(am)).to(device),
        torch.from_numpy(np.stack(gv)).to(device),
    )


def curriculum_sample(words: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample n words: 35% short (≤6), 35% medium (7–10), 30% long (≥11)."""
    short  = [w for w in words if len(w) <= 6]
    medium = [w for w in words if 7 <= len(w) <= 10]
    long_  = [w for w in words if len(w) >= 11]

    n_short  = round(n * 0.35)
    n_medium = round(n * 0.35)
    n_long   = n - n_short - n_medium

    def _pick(pool: list[str], k: int) -> list[str]:
        # Fall back to full word list if a length bucket is empty
        return rng.choices(pool if pool else words, k=k)

    batch = _pick(short, n_short) + _pick(medium, n_medium) + _pick(long_, n_long)
    rng.shuffle(batch)
    return batch


# Rare letters that are systematically under-predicted by supervised training
_RARE_LETTERS = set("jkwzqx")


def targeted_sample(words: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample n words: 80% natural + 20% forced from rare-letter (j/k/w/z/q/x) words.

    Concentrates the RL gradient on the letter-bias gap identified in exp_k:
    j (−39pp), k (−27pp), w (−16pp), z (−11pp) below average win rate.
    """
    rare_words = [w for w in words if any(c in _RARE_LETTERS for c in w)]
    n_rare   = round(n * 0.20)
    n_normal = n - n_rare
    batch = (rng.sample(words, n_normal)
             + rng.choices(rare_words if rare_words else words, k=n_rare))
    rng.shuffle(batch)
    return batch


def compute_aux_loss(
    model: HangmanPolicy,
    words: list[str],
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    """
    Auxiliary supervised BCE loss on random hangman states from the batch words.
    Prevents catastrophic forgetting of letter-membership knowledge from Stage 1.
    """
    states = [sample_state(w, rng) for w in words]
    tok_list, am_list, gv_list, tgt_list, lm_list = [], [], [], [], []
    for s in states:
        ids, am = encode_masked(s.masked, MAX_WORD_LEN)
        gv      = encode_guessed(s.guessed)
        tgt, lm = encode_target(s)
        tok_list.append(ids); am_list.append(am); gv_list.append(gv)
        tgt_list.append(tgt); lm_list.append(lm)

    tok = torch.from_numpy(np.stack(tok_list)).to(device)
    am  = torch.from_numpy(np.stack(am_list)).to(device)
    gv  = torch.from_numpy(np.stack(gv_list)).to(device)
    tgt = torch.from_numpy(np.stack(tgt_list)).to(device)
    lm  = torch.from_numpy(np.stack(lm_list)).to(device)

    logits = model(tok, am, gv)
    bce    = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * lm).sum() / lm.sum().clamp(min=1)


def collect_rollout(
    model: HangmanPolicy,
    words: list[str],
    device: torch.device,
    win_bonus: float,
    loss_penalty: float,
    terminal_reward: bool = False,
    proportional_reward: bool = False,
):
    """
    Play one game per word with the current policy (stochastic, no gradient).
    Stores every (state, action, old_logp) for PPO replay across K epochs.

    Returns flat tensors over all steps in all games:
      flat_tok    [N, max_len]  token ids
      flat_am     [N, max_len]  attention mask
      flat_gv     [N, 26]       guessed vector
      flat_act    [N]            action index
      flat_lp_old [N]            log π_old(a|s)  — detached
      flat_ret    [N]            return-to-go (γ=1)
      wins        [B] bool
    Returns None if all games produced empty trajectories.
    """
    games        = [HangmanGame(w) for w in words]
    traj_tok     = [[] for _ in games]
    traj_am      = [[] for _ in games]
    traj_gv      = [[] for _ in games]
    traj_act     = [[] for _ in games]
    traj_lp      = [[] for _ in games]
    step_rewards = [[] for _ in games]
    wins         = [False] * len(games)

    with torch.no_grad():
        for _t in range(26):
            active = [i for i, g in enumerate(games) if not g.done]
            if not active:
                break

            tok, am, gv = encode_batch_states([games[i] for i in active], device)
            logits    = model(tok, am, gv)
            logits    = logits.masked_fill(gv.bool(), float("-inf"))
            logp      = F.log_softmax(logits, dim=-1)
            probs     = logp.exp()
            actions   = torch.multinomial(probs, num_samples=1).squeeze(-1)
            chosen_lp = logp.gather(1, actions.unsqueeze(1)).squeeze(1)

            for slot, i in enumerate(active):
                traj_tok[i].append(tok[slot].cpu())
                traj_am[i].append(am[slot].cpu())
                traj_gv[i].append(gv[slot].cpu())
                traj_act[i].append(int(actions[slot].item()))
                traj_lp[i].append(chosen_lp[slot].item())

                letter = chr(ord("a") + traj_act[i][-1])
                # count revealed positions before guess for proportional reward
                if proportional_reward:
                    before_revealed = sum(c != "_" for c in games[i].masked)
                hit    = games[i].guess(letter)
                if terminal_reward:
                    step_rewards[i].append(0.0)   # no step reward; terminal only
                elif proportional_reward:
                    wlen = len(games[i].word)
                    if hit:
                        after_revealed = sum(c != "_" for c in games[i].masked)
                        step_rewards[i].append((after_revealed - before_revealed) / wlen)
                    else:
                        step_rewards[i].append(-1.0 / wlen)
                else:
                    step_rewards[i].append(1.0 if hit else -1.0)

    for i, g in enumerate(games):
        if not step_rewards[i]:
            continue
        if g.won:
            wins[i] = True
            if terminal_reward:
                step_rewards[i][-1] = 1.0        # win = +1 (overwrite)
            elif proportional_reward:
                step_rewards[i][-1] += 1.0        # win bonus = +1 on top of step reward
            else:
                step_rewards[i][-1] += win_bonus
        else:
            if terminal_reward:
                step_rewards[i][-1] = -1.0        # loss = -1 (overwrite)
            elif proportional_reward:
                step_rewards[i][-1] += -1.0       # loss penalty = −1 on top of step reward
            else:
                step_rewards[i][-1] += loss_penalty

    all_tok, all_am, all_gv, all_act, all_lp, all_ret, all_game_idx = [], [], [], [], [], [], []
    game_total_rets: list[float] = []   # one entry per game (0.0 for empty trajectories)

    for i in range(len(games)):
        r = step_rewards[i]
        game_total_rets.append(sum(r) if r else 0.0)
        if not r:
            continue
        G, rtg = 0.0, []
        for rew in reversed(r):
            G = rew + G
            rtg.append(G)
        rtg.reverse()
        for t in range(len(r)):
            all_tok.append(traj_tok[i][t])
            all_am.append(traj_am[i][t])
            all_gv.append(traj_gv[i][t])
            all_act.append(traj_act[i][t])
            all_lp.append(traj_lp[i][t])
            all_ret.append(rtg[t])
            all_game_idx.append(i)      # original game index — needed for GRPO grouping

    if not all_tok:
        return None

    return (
        torch.stack(all_tok).to(device),
        torch.stack(all_am).to(device),
        torch.stack(all_gv).to(device),
        torch.tensor(all_act,      dtype=torch.long,    device=device),
        torch.tensor(all_lp,       dtype=torch.float32, device=device),
        torch.tensor(all_ret,      dtype=torch.float32, device=device),
        torch.tensor(all_game_idx, dtype=torch.long,    device=device),
        game_total_rets,
        wins,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init",           required=True, help="Path to supervised checkpoint")
    ap.add_argument("--steps",          type=int,   default=8_000)
    ap.add_argument("--games-per-step", type=int,   default=512)
    ap.add_argument("--lr",             type=float, default=1e-5)
    ap.add_argument("--clip-eps",       type=float, default=0.2,
                    help="PPO clip range ε — prevents destructive policy updates (default: 0.2)")
    ap.add_argument("--ppo-epochs",     type=int,   default=4,
                    help="Optimisation epochs per collected rollout batch (default: 4)")
    ap.add_argument("--entropy-coef",   type=float, default=0.003)
    ap.add_argument("--aux-coef",       type=float, default=0.1,
                    help="Weight on auxiliary supervised BCE loss (default: 0.1)")
    ap.add_argument("--kl-coef",        type=float, default=0.0,
                    help="KL-divergence penalty weight: kl_coef * KL(π_θ || π_ref) added to loss. "
                         "Uses a frozen copy of --init (or --kl-ref) as the reference model. "
                         "RLHF-style regularisation: prevents forgetting without aux BCE equilibrium. "
                         "Recommended: set --aux-coef 0 when using --kl-coef. (default: 0.0 = disabled)")
    ap.add_argument("--kl-ref",         default=None,
                    help="Path to reference checkpoint for KL penalty (default: same as --init). "
                         "Only used when --kl-coef > 0.")
    ap.add_argument("--win-bonus",       type=float, default=5.0)
    ap.add_argument("--loss-penalty",   type=float, default=-2.0)
    ap.add_argument("--terminal-reward", action="store_true",
                    help="Use terminal-only reward: +1 win / -1 loss. Disables step +1/-1 rewards.")
    ap.add_argument("--proportional-reward", action="store_true",
                    help="Step reward = positions_revealed/word_len (hit) or -1/word_len (miss), "
                         "+1/-1 terminal bonus. Normalises reward by word length.")
    ap.add_argument("--targeted-sampling", action="store_true",
                    help="20%% of each batch oversampled from rare-letter words (j/k/w/z/q/x) "
                         "to concentrate gradient on the letter-bias gap.")
    ap.add_argument("--grpo-k",         type=int, default=1,
                    help="GRPO group size: play K games per unique word and normalise advantages "
                         "relative to the per-word mean return, removing word-difficulty confound. "
                         "games-per-step must be divisible by K. (default: 1 = standard PPO)")
    ap.add_argument("--ppo-minibatch",  type=int, default=4096,
                    help="Max rows per mini-batch inside PPO epoch (default: 4096). "
                         "Reduces peak VRAM when games-per-step is large.")
    ap.add_argument("--curriculum",     action="store_true",
                    help="Sample 35%% short / 35%% medium / 30%% long words per batch")
    ap.add_argument("--eval-every",     type=int,   default=100)
    ap.add_argument("--log-every",      type=int,   default=20)
    ap.add_argument("--val-subset",     type=int,   default=1_500)
    ap.add_argument("--device",         default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--demo-every",     type=int,   default=100,
                    help="Play and log a demo game every N steps (0 = off)")
    ap.add_argument("--ckpt-dir",       default="checkpoints")
    ap.add_argument("--log-file",       default=None)
    ap.add_argument("--save-every",     type=int, default=0,
                    help="Save a periodic checkpoint rl_stepN.pt every N steps (0 = off). "
                         "Enables post-hoc test eval at each checkpoint to find true best.")
    args = ap.parse_args()

    logger = setup_logging(args.log_file)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    logger.info(f"[device] {device}")

    ckpt  = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg   = ModelConfig(**ckpt["cfg"])
    model = HangmanPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"[init] supervised checkpoint  val={ckpt.get('val_win_rate', 0)*100:.2f}%")

    # KL reference model — frozen copy of supervised init (or explicit --kl-ref path).
    # Used by --kl-coef to regularise the policy toward the supervised checkpoint,
    # preventing catastrophic forgetting without the degraded-equilibrium problem
    # of aux BCE (which creates an independent 58-59% floor via random-state BCE).
    ref_model = None
    if args.kl_coef > 0.0:
        ref_path = args.kl_ref if args.kl_ref else args.init
        ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False) \
                   if ref_path != args.init else ckpt
        ref_model = HangmanPolicy(cfg).to(device)
        ref_model.load_state_dict(ref_ckpt["model_state"])
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        logger.info(f"[kl]   frozen ref model loaded  kl_coef={args.kl_coef}  "
                    f"ref={ref_path}")

    logger.info(
        f"[ppo]  clip_eps={args.clip_eps}  ppo_epochs={args.ppo_epochs}  "
        f"aux_coef={args.aux_coef}  kl_coef={args.kl_coef}  "
        f"curriculum={args.curriculum}  "
        f"terminal_reward={args.terminal_reward}  proportional_reward={args.proportional_reward}  "
        f"targeted_sampling={args.targeted_sampling}  grpo_k={args.grpo_k}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_words = load_word_list("data/train_words.txt")
    val_words   = load_word_list("data/val_words.txt")
    val_eval    = val_words[: args.val_subset]
    logger.info(f"[data] train={len(train_words):,}  val_eval={len(val_eval):,}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = ckpt.get("val_win_rate", 0.0)
    win_ma   = 0.0
    t_start  = time.time()

    for step in range(1, args.steps + 1):

        # ── 1. sample batch ───────────────────────────────────────────────────
        if args.grpo_k > 1:
            # GRPO: sample unique words, then repeat each K times so games are grouped by word.
            # batch = [w0, w0, ...(K), w1, w1, ...(K), ...]
            n_unique = args.games_per_step // args.grpo_k
            unique_words = rng.sample(train_words, n_unique)
            batch = [w for w in unique_words for _ in range(args.grpo_k)]
        elif args.curriculum:
            batch = curriculum_sample(train_words, args.games_per_step, rng)
        elif args.targeted_sampling:
            batch = targeted_sample(train_words, args.games_per_step, rng)
        else:
            batch = rng.sample(train_words, args.games_per_step)

        # ── 2. collect rollout (no gradient) ──────────────────────────────────
        model.train()
        rollout = collect_rollout(model, batch, device, args.win_bonus, args.loss_penalty,
                                  terminal_reward=args.terminal_reward,
                                  proportional_reward=args.proportional_reward)
        if rollout is None:
            continue
        flat_tok, flat_am, flat_gv, flat_act, flat_lp_old, flat_ret, flat_game_idx, game_total_rets, wins = rollout

        # ── 3. normalise advantages ────────────────────────────────────────────
        if args.grpo_k > 1:
            # GRPO: subtract per-word mean return so word difficulty cancels out.
            # batch is [w0]*K + [w1]*K + ..., so game i → word i // grpo_k.
            # game_total_rets[i] = total episode return for game i.
            n_unique = len(batch) // args.grpo_k
            game_rets_t = torch.tensor(game_total_rets, dtype=torch.float32, device=device)
            # shape [n_unique, grpo_k] → mean over K games per word → [n_unique]
            word_baselines = game_rets_t.view(n_unique, args.grpo_k).mean(dim=1)
            # map each trajectory step to its word's baseline
            step_word_idx      = flat_game_idx // args.grpo_k        # [N_steps]
            per_step_baseline  = word_baselines[step_word_idx]       # [N_steps]
            baseline           = flat_ret.mean()                     # for logging only
            flat_adv = ((flat_ret - per_step_baseline) / flat_ret.std().clamp(min=1e-3)).detach()
        else:
            # Standard PPO: single batch-level baseline
            baseline = flat_ret.mean()
            flat_adv = ((flat_ret - baseline) / flat_ret.std().clamp(min=1e-3)).detach()

        win_rate = sum(wins) / len(wins)
        win_ma   = 0.95 * win_ma + 0.05 * win_rate

        # ── 4. PPO optimisation epochs ─────────────────────────────────────────
        # Mini-batch size caps peak activation memory during backward pass.
        # A 256-row mini-batch through a 4-layer d=256 transformer uses ~150 MB
        # activations — safe on any GPU that can hold the model itself.
        ppo_loss_last = torch.tensor(0.0)
        aux_loss_last = torch.tensor(0.0)
        kl_loss_last  = torch.tensor(0.0)
        n_steps = len(flat_tok)
        mb = args.ppo_minibatch  # max rows per mini-batch

        # Aux loss words: cap to 256 to keep its backward pass small
        aux_words = rng.sample(batch, min(256, len(batch)))

        # Pre-compute reference model log-probs at rollout states (no grad, one pass).
        # Shape [N_steps, 26]. Done once per training step (not per PPO epoch) for efficiency.
        ref_logp_flat = None
        if ref_model is not None:
            with torch.no_grad():
                ref_logits = ref_model(flat_tok, flat_am, flat_gv)
                ref_logits = ref_logits.masked_fill(flat_gv.bool(), float("-inf"))
                ref_logp_flat = F.log_softmax(ref_logits, dim=-1)  # [N, 26]

        for _epoch in range(args.ppo_epochs):
            # Shuffle steps so each epoch sees a different order
            perm = torch.randperm(n_steps, device=device)

            # Pass 1: PPO + KL gradient — mini-batched
            opt.zero_grad(set_to_none=True)
            ppo_acc = 0.0
            kl_acc  = 0.0

            for mb_start in range(0, n_steps, mb):
                idx     = perm[mb_start : mb_start + mb]
                mb_size = len(idx)
                scale   = mb_size / n_steps  # weight for gradient accumulation

                logits  = model(flat_tok[idx], flat_am[idx], flat_gv[idx])
                logits  = logits.masked_fill(flat_gv[idx].bool(), float("-inf"))
                logp    = F.log_softmax(logits, dim=-1)
                new_lp  = logp.gather(1, flat_act[idx].unsqueeze(1)).squeeze(1)

                ratio   = torch.exp(new_lp - flat_lp_old[idx])
                adv     = flat_adv[idx]
                clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                ppo_loss = -torch.min(ratio * adv, clipped * adv).mean()

                finite  = torch.isfinite(logp)
                safe_p  = torch.where(finite, logp.exp(), torch.zeros_like(logp))
                safe_lp = torch.where(finite, logp,       torch.zeros_like(logp))
                entropy = -(safe_p * safe_lp).sum(dim=-1).mean()

                mb_loss = ppo_loss - args.entropy_coef * entropy

                # KL(π_θ || π_ref) at rollout states — RLHF-style regularisation.
                # KL = sum_a π_θ(a) * [log π_θ(a) − log π_ref(a)].
                # Safe: guessed letters have curr_prob=0 AND ref_logp=-inf.
                # Guard the difference where ref_logp=-inf to avoid 0 * inf = NaN.
                if ref_model is not None and args.kl_coef > 0.0:
                    curr_probs = safe_p                          # [mb, 26]
                    ref_lp_mb  = ref_logp_flat[idx]             # [mb, 26]
                    safe_diff  = torch.where(
                        torch.isfinite(ref_lp_mb),
                        safe_lp - ref_lp_mb,
                        torch.zeros_like(ref_lp_mb),
                    )
                    kl_per  = (curr_probs * safe_diff).sum(dim=-1)  # [mb]
                    kl_loss = kl_per.mean()
                    mb_loss = mb_loss + args.kl_coef * kl_loss
                    kl_acc += kl_loss.item() * scale

                (mb_loss * scale).backward()
                ppo_acc += ppo_loss.item() * scale

            # Pass 2: aux supervised loss — separate backward, same opt step
            # (kept separate so its activations don't overlap with PPO activations)
            if args.aux_coef > 0:
                aux_loss = compute_aux_loss(model, aux_words, rng, device)
                (args.aux_coef * aux_loss).backward()
                aux_loss_last = aux_loss.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ppo_loss_last = torch.tensor(ppo_acc)
            kl_loss_last  = torch.tensor(kl_acc)

        # free rollout tensors + fragmented cache before next step
        del flat_tok, flat_am, flat_gv, flat_act, flat_lp_old, flat_ret, flat_adv, flat_game_idx
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"[ppo step {step:>5}/{args.steps}]  "
                f"win(batch)={win_rate*100:5.1f}%  win_ma={win_ma*100:5.1f}%  "
                f"ret={baseline.item():+.2f}  "
                f"ppo={ppo_loss_last.item():+.4f}  aux={aux_loss_last.item():.4f}  "
                f"kl={kl_loss_last.item():.4f}  "
                f"elapsed={elapsed:6.1f}s"
            )

        if args.demo_every > 0 and step % args.demo_every == 0:
            word = rng.choice(val_words)
            logger.info(f"[demo ppo step={step}] ── greedy game on '{word}' ──────────────")
            demo_game(model, word, device, logger)

        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__,
                 "step": step, "val_win_rate": best_val},
                ckpt_dir / f"rl_step{step}.pt",
            )
            logger.info(f"  → periodic checkpoint saved: rl_step{step}.pt")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_win_rate(model, val_eval, device)
            wr = metrics["win_rate"]
            logger.info(f"[eval ppo step={step}]  val_win_rate={wr*100:.2f}%")
            if wr > best_val:
                best_val = wr
                torch.save(
                    {"model_state": model.state_dict(), "cfg": cfg.__dict__,
                     "step": step, "val_win_rate": wr},
                    ckpt_dir / "rl_best.pt",
                )
                logger.info(f"  → saved rl_best.pt  (val={wr*100:.2f}%)")

    logger.info(f"\n[done ppo] best val win-rate: {best_val*100:.2f}%")


if __name__ == "__main__":
    main()
