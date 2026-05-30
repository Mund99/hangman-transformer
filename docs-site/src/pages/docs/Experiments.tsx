import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, Diagram, Badge } from '../../framework'

export default function Experiments() {
  return (
    <DocLayout>
      <h1>Experiments &amp; Findings</h1>
      <div className="page-meta">
        <Badge type="active" label="20+ runs" />
        <Badge type="archived" label="best: 69.02%" />
      </div>

      <p>
        This is the full record of what was tried, in order, with the result and the reasoning
        behind each. The headline is simple: <strong>architecture scaling worked, everything
        else failed</strong> — and the failures are the interesting part.
      </p>

      <Callout type="quote" title="How the experiments were run">
        Every run started from a <em>specific hypothesis</em> and the binary question it answered —
        not a hyperparameter sweep. The guiding principle: “RL solved AlphaGo; Hangman isn’t hard.
        What’s needed is critical thinking about <em>why</em> something fails, not more sweeps.”
        Progress was always measured by validation win rate on held-out words, never by training
        loss or demo games.
      </Callout>

      <Diagram>{`graph TD
    A["sup_search 1–4<br/>step + data scaling<br/>→ 66.20% ceiling at d=256"] --> B["sup_finetune<br/>rare-letter reweighting<br/>→ 66.32% (+0.12pp)"]
    B --> C["rl_search 1–6<br/>PPO · KL-PPO · GRPO<br/>→ all failed, max 66.22%"]
    C --> D["sup_search5<br/>scale to d=384, d=512<br/>→ 68.79%"]
    D --> E["sup_search6<br/>16×/4× curriculum<br/>→ 65.78% REGRESSION"]
    E --> F["inference search<br/>7 variants<br/>→ all regressed"]
    F --> G["sup_search7<br/>dedicated short-word model<br/>→ regression"]
    G --> H["sup_search8<br/>d=768 + d=1024<br/>→ 69.02% BEST · d=1024 regressed"]`}</Diagram>

      <h2>Leaderboard</h2>
      <DocTable
        headers={['Rank', 'Model', 'Test %', 'Params', 'Notes']}
        rows={[
          ['🥇', 'sup8_a — d=768', '69.02%', '43.2M', 'Best overall'],
          ['🥈', 'sup5_b — d=512', '68.79%', '19.2M', 'Powers the in-browser demo'],
          ['🥉', 'sup5_a — d=384', '68.24%', '10.8M', ''],
          ['4', 'sup8_b — d=1024', '67.69%', '76.8M', 'Regression — corpus too small'],
          ['5', 'ft_b — d=256', '66.32%', '3.25M', 'Rare-letter reweighting'],
          ['6', 'rl5_b — KL-PPO', '66.22%', '3.25M', 'Best RL run — still below supervised'],
        ]}
      />

      <h2>1 · Architecture scaling — the one thing that worked</h2>
      <p>
        Holding everything else fixed and varying only model width revealed a clean logarithmic
        scaling law. Each doubling of capacity bought roughly half the previous gain — until
        d=1024, where the model regressed.
      </p>

      <DocTable
        headers={['d_model', 'Layers', 'Params', 'Test %', 'Δ from prev']}
        rows={[
          ['256', '4', '3.25M', '66.20%', 'baseline'],
          ['384', '6', '10.8M', '68.24%', '+2.04pp (3.3× params)'],
          ['512', '6', '19.2M', '68.79%', '+0.55pp (1.8× params)'],
          ['768  ★', '6', '43.2M', '69.02%', '+0.23pp (2.3× params)'],
          ['1024', '6', '76.8M', '67.69%', '−1.10pp — regression'],
        ]}
      />

      <Callout type="info" title="Finding 1 + 8 — capacity helps until the data runs out">
        d=768 is the sweet spot. At d=1024 (76.8M params) the model regressed by 1.10pp: 317k
        words simply cannot support that many parameters. The corpus, not the architecture, is
        now the ceiling.
      </Callout>

      <h2>2 · Reinforcement learning — six failures</h2>
      <p>
        RL was the original hypothesis: let the model play full games and learn winning strategy
        from rewards. It never worked. All six searches — vanilla PPO, KL-penalised PPO at
        several strengths, and GRPO — plateaued at or below the supervised baseline.
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

      <Callout type="warning" title="Finding 2 — Hangman is a knowledge task, not a strategy task">
        RL learns strategy through experience. But optimal Hangman play is almost entirely about
        <em> knowing English letter statistics</em> — which supervised learning captures directly
        and completely. There is no strategic layer left for RL to add, so the KL penalty that
        kept it stable also kept it from changing anything useful.
      </Callout>

      <h2>3 · The short-word problem — structural, not fixable</h2>
      <p>
        Short words (4–6 letters) are the model’s weak spot: a 4-letter word offers so few
        positional constraints that even a perfect frequency model often can’t resolve it. Two
        separate attacks both failed.
      </p>

      <h3>Aggressive curriculum (sup_search6)</h3>
      <DocTable
        headers={['Curriculum', 'Test %', 'Effect']}
        rows={[
          ['4×/2× (baseline)', '68.79%', 'Validated sweet spot'],
          ['16×/4×', '65.78%', '−3.01pp — long words starved'],
        ]}
      />

      <h3>Dedicated short-word model (sup_search7)</h3>
      <DocTable
        headers={['Model', 'Short-word win %', 'Note']}
        rows={[
          ['Full model (all 317k words)', '25.77%', 'Best on short words'],
          ['Dedicated (48k short words only)', '19–22%', 'Worse, despite specialising'],
          ['Ensemble (short + full)', '−0.6 to −0.9pp', 'Every variant regressed'],
        ]}
      />

      <Callout type="warning" title="Finding 3 + 5 + 7 — specialisation hurts">
        Training a model only on short words made it <em>worse</em> at short words. Long words
        carry patterns — suffixes, consonant clusters, positional n-grams — that transfer down.
        Remove that signal and you lose more than you gain. The 4×/2× curriculum is a genuine
        sweet spot, not a floor: pushing harder always breaks the long/short balance.
      </Callout>

      <h2>4 · Inference-time search — a tempting trap</h2>
      <p>
        At test time you can filter the training dictionary to words matching the current pattern
        and use that pool’s letter frequencies. It sounds like free accuracy. All seven variants
        regressed.
      </p>

      <DocTable
        headers={['Strategy', 'Test %', 'Δ vs greedy']}
        rows={[
          ['Greedy (model argmax)', '68.79%', 'baseline'],
          ['Hybrid α=0.7', '67.75%', '−1.04pp'],
          ['Adaptive-k', '66.8–67.1%', '−1.7 to −2pp'],
          ['Frequency-only', '64.46%', '−4.33pp'],
        ]}
      />

      <Callout type="danger" title="Finding 6 — out-of-vocabulary breaks candidate pools">
        Test words are 100% disjoint from training. The candidate pool reflects the
        <em> training</em> distribution, which biases guesses toward common patterns — exactly
        wrong for the unusual held-out words. The model’s globally-learned statistics are a better
        prior than any local pool. (Candidate search only wins when the pool contains the actual
        answer set, as in Wordle.)
      </Callout>

      <h2>5 · Rare letters — reweighting barely moved them</h2>
      <p>
        The letters <code>j</code>, <code>k</code>, <code>w</code> have the lowest per-letter win
        rates. Two reweighting strategies were tried: oversampling words containing rare letters,
        and up-weighting the BCE loss on those letters. The best result (10× loss weight,
        low LR) nudged the model from 66.20% → <strong>66.32%</strong> — a +0.12pp gain.
      </p>

      <Callout type="info" title="Finding 4 — rare-letter weakness is intrinsic">
        <code>j/k/w</code> are weak because they are genuinely rare in English, so each carries
        little positional signal. This is a property of the language, not a training bug — and it
        barely responds to reweighting.
      </Callout>

      <h2>6 · Smaller findings</h2>
      <DocTable
        headers={['Finding', 'Detail']}
        rows={[
          ['Data cleaning ≈ noise', 'Removing 248 / 317k garbage words moved results < 0.1pp'],
          ['Val vs test gap is normal', 'd=768: val 70.52% vs test 69.02% — checkpoint-selection bias, not overfitting'],
          ['int8 quantization is free', 'Browser model matches full precision within noise'],
          ['Greedy decoding is optimal', 'Beam search and sampling at inference both regressed'],
        ]}
      />

      <Callout type="note" title="How results were measured">
        Every win rate is greedy argmax play on a held-out set: the full 17,652-word test set for
        final numbers, and a 2,000-word validation subset for fast in-training checkpoints. No
        shaped rewards or demo games count toward the reported metrics.
      </Callout>

      <h2>How this compares to published Hangman agents</h2>
      <p>
        Context matters: a 69% supervised result is strong relative to public RL-based Hangman
        agents, which tend to be far weaker or report inflated numbers.
      </p>

      <DocTable
        headers={['Source', 'Approach', 'Reported', 'Reality check']}
        rows={[
          ['This project', 'Supervised Transformer', '69.02%', 'Greedy, 17,652 held-out words'],
          ['Sagar Nildas', 'PPO-LSTM', '“65%”', 'Actual win rate 15.2%; “65” was a shaped-reward value, easier 12-life game'],
          ['ZosoV', 'BERT + DQN', '~65%', 'Different 113k corpus; the BERT prior likely does the work, not the DQN'],
        ]}
      />

      <Callout type="tip" title="The bigger lesson">
        No public RL Hangman agent convincingly beats a well-trained supervised model by RL alone.
        Our plain supervised baseline (66.2%) is already ~4× the best verifiable public RL result
        (15.2%). For knowledge-bound tasks, supervision is the workhorse — RL is not a free upgrade.
      </Callout>

      <h2>Where it lands</h2>
      <Callout type="tip" title="Final result: 69.02% (d=768, 43.2M params)">
        Every training-side, inference-side, and specialisation idea was exhausted. The practical
        ceiling for character-level supervised Hangman on this 317k-word corpus is ~69%. Closing
        the gap to perfect play (~75–80%) would need a much larger corpus or a fundamentally
        different approach — e.g. an LLM with built-in language priors.
      </Callout>

      <p>
        See the model and the scaling law up close on the{' '}
        <Link to="/">Overview</Link>, or <Link to="/play">watch it play</Link>.
      </p>
    </DocLayout>
  )
}
