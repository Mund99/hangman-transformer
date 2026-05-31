import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, CodeBlock, Diagram, Badge } from '../../framework'

export default function Training() {
  return (
    <DocLayout>
      <h1>Training: Supervised &amp; RL</h1>
      <div className="page-meta">
        <Badge type="active" label="supervised: works" />
        <Badge type="archived" label="RL: failed ×6" />
      </div>

      <p>
        The project name says “RL,” but the honest finding is that{' '}
        <strong>supervised learning did all the work</strong> and reinforcement learning never
        improved on it. This page covers both — the method that succeeded and the one that
        didn’t, with the reason why.
      </p>

      <Diagram>{`graph LR
    A["Supervised pretraining<br/>predict letters in the word"] -->|68–69%| B["Best model"]
    A -->|init| C["RL fine-tuning<br/>play games, reward wins"]
    C -->|no gain| B`}</Diagram>

      <h2>Supervised training — the method that worked</h2>
      <p>
        The model is shown a random mid-game board (see{' '}
        <Link to="/docs/data">state sampling</Link>) and trained to predict which letters appear
        in the hidden word. It never plays a full game during training — just classifies static
        snapshots — yet this is enough to reach a 69% win rate at play time.
      </p>

      <CodeBlock language="python" filename="training step">{`logits = model(token_ids, attn_mask, guessed_vec)        # [B, 26]
# multi-label BCE, only on letters not yet guessed
loss = bce_with_logits(logits, target, weight=loss_mask)
loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()   # AdamW, cosine LR with warmup`}</CodeBlock>

      <DocTable
        headers={['Choice', 'Setting', 'Why']}
        rows={[
          ['Loss', 'Multi-label BCE', 'A word has several letters at once — not one class'],
          ['Optimizer', 'AdamW + cosine LR', 'Stable convergence over 200k steps'],
          ['Length curriculum', '4× short / 2× medium', 'Oversample hard short words (validated sweet spot)'],
          ['Checkpoint', 'Best validation win rate', 'Selected by games won, not by loss'],
        ]}
      />

      <h3>The target and the loss mask</h3>
      <p>
        For each board, two 26-dim vectors are built. The <strong>target</strong> marks every
        letter that appears anywhere in the word; the <strong>loss mask</strong> zeroes out letters
        that have already been guessed, so the model is only graded on decisions it still has to
        make.
      </p>

      <CodeBlock language="python">{`# word = "apple", guessed = {a, l, s}
target    = [1 if letter in set(word) else 0 for letter in "a..z"]   # a,p,l,e -> 1
loss_mask = [0 if letter in guessed     else 1 for letter in "a..z"] # a,l,s   -> 0
# loss is computed only where loss_mask == 1 (here: p and e must be predicted)`}</CodeBlock>

      <Callout type="tip" title="Why multi-label BCE, not 26-way softmax">
        A softmax forces the model to pick <em>one</em> letter — but <code>apple</code> contains{' '}
        <code>a, p, l, e</code> all at once. Binary cross-entropy treats each of the 26 letters as
        an independent yes/no question, which matches the real structure: “is this letter in the
        word?” asked 26 times in parallel.
      </Callout>

      <Callout type="tip" title="The length curriculum">
        Short words (4–6 letters) are rare and hard, so they’re oversampled 4× (medium 2×) during
        training. This 4×/2× ratio is a genuine sweet spot: weaker, and short words are
        undertrained; stronger (e.g. 16×/4×), and long words starve — a −3pp regression. See{' '}
        <Link to="/docs/experiments">Experiments</Link>.
      </Callout>

      <Callout type="note" title="Val vs test gap is not overfitting">
        The best model scores val 70.52% vs test 69.02%. That ~1.5pp gap is expected: the
        checkpoint is <em>chosen</em> by validation score, so validation is mildly optimistic.
        Both sets are fully held out from training.
      </Callout>

      <h2>RL fine-tuning — and why it failed</h2>
      <p>
        The original plan was textbook: pretrain with supervision, then fine-tune with
        reinforcement learning so the model could optimise directly for <em>winning games</em>{' '}
        rather than <em>predicting letters</em>. Six separate searches were run. None beat the
        supervised baseline.
      </p>

      <DocTable
        headers={['Search', 'Algorithm', 'Result']}
        rows={[
          ['rl_search1', 'PPO', 'Degraded immediately'],
          ['rl_search2–3', 'KL-PPO (kl 0.01, 0.05)', 'Plateaued at baseline'],
          ['rl_search4', 'KL-PPO (kl 0.10)', '66.22% — best RL, still ≤ supervised'],
          ['rl_search5', 'KL-PPO (kl 1.0)', '66.19%'],
          ['rl_search6', 'GRPO', 'Collapsed'],
        ]}
      />

      <h3>The reward and the objective</h3>
      <CodeBlock language="python">{`# per-step rewards during a played game
+1  correct guess        -1  wrong guess
+5  win bonus            -2  loss penalty

# KL-PPO objective — clip + stay close to the supervised prior
loss = ppo_clip(ratio, advantage) + kl_coef * KL(pi_rl || pi_supervised)`}</CodeBlock>

      <h3>How, exactly, each search failed</h3>
      <p>
        The failures weren’t random — each search fixed the previous one’s flaw and hit a new
        wall, all bottoming out at the same 58–59% floor (on the d=256 model).
      </p>

      <DocTable
        headers={['Search', 'Setup', 'Why it failed']}
        rows={[
          ['1', 'Shaped reward, batch advantage', 'Curriculum + 4 PPO epochs + small batch → stale, noisy updates'],
          ['2', 'Terminal-only reward', 'Reward too sparse → gradient ≈ 0, nothing to learn from'],
          ['3', 'Shaped reward, fixed config', 'Batch-norm advantage conflates word difficulty with action quality'],
          ['4', 'GRPO grouped advantage', 'Auxiliary BCE loss pins the policy at a degraded equilibrium'],
          ['5–6', 'KL-PPO (RLHF-style)', 'Validation nudged up, but never transferred to the test set'],
        ]}
      />

      <Callout type="warning" title="The most revealing failure">
        With batch-level advantage normalization, easy words return ≈ +8–10 and hard
        (rare-letter) words ≈ −5 to −8. The policy “learns” that rare letters lead to bad outcomes
        and starts <em>avoiding</em> them — so the j/k/w letter bias gets <strong>worse</strong>,
        not better. RL optimised the policy in the wrong direction while the loss happily fell.
      </Callout>

      <Callout type="danger" title="Why RL can’t help here: it’s a knowledge task">
        Underneath every mechanism is one fact: optimal Hangman play is about{' '}
        <em>knowing English letter statistics</em>, which supervised learning already captures
        directly. There’s no strategic layer left for RL to discover. The KL penalty that kept
        training stable also kept it from changing anything useful; remove it and the model forgets
        its knowledge and degrades. The real lever was never the RL algorithm — it was{' '}
        <Link to="/docs/architecture">model capacity</Link>.
      </Callout>

      <h3>The takeaway</h3>
      <p>
        When RL refuses to improve on a supervised model, it’s often a signal that the model
        already has the right inductive bias and the task has little hidden strategy. The right
        next move was more capacity, not a better RL algorithm — which is exactly what the{' '}
        <Link to="/docs/architecture">scaling experiments</Link> confirmed.
      </p>

      <p>
        The full run-by-run record is on the{' '}
        <Link to="/docs/experiments">Experiments &amp; Findings</Link> page.
      </p>
    </DocLayout>
  )
}
