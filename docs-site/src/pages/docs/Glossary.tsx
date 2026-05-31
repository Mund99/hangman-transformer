import { DocLayout, DocTable, Badge } from '../../framework'

export default function Glossary() {
  return (
    <DocLayout>
      <h1>Glossary</h1>
      <div className="page-meta">
        <Badge type="active" label="terms" />
      </div>

      <p>Short definitions for the terms used across these docs.</p>

      <h2>Model &amp; architecture</h2>
      <DocTable
        headers={['Term', 'Meaning']}
        rows={[
          ['Transformer', 'Neural architecture where every position attends to every other via scaled dot-product attention — no recurrence.'],
          ['Encoder', 'A Transformer that reads the whole input at once (bidirectional). The opposite of a decoder, which generates left to right.'],
          ['Self-attention', 'Mechanism letting each position weigh and aggregate information from all others.'],
          ['d_model', 'The model’s hidden width. The main capacity knob — 256 → 768 in this project.'],
          ['Pre-norm', 'LayerNorm applied inside the residual branch (norm_first=True) — keeps gradients stable in deeper/wider models.'],
          ['GELU', 'A smooth activation function used in the feed-forward and head layers.'],
          ['Masked mean pooling', 'Averaging the encoder outputs over only the real (non-padding) positions to get one word-level vector.'],
          ['Logit', 'A raw, pre-sigmoid score — here, one per letter.'],
        ]}
      />

      <h2>Training</h2>
      <DocTable
        headers={['Term', 'Meaning']}
        rows={[
          ['BCE (binary cross-entropy)', 'Loss treating each letter as an independent yes/no prediction — fits “which letters are in the word”.'],
          ['Multi-label', 'A prediction where several outputs can be “true” at once (a, p, l, e in “apple”), unlike single-class softmax.'],
          ['Loss mask', 'Zeroes the loss on already-guessed letters, so the model is only trained on decisions still to make.'],
          ['Curriculum / WeightedRandomSampler', 'Oversampling harder short words (4× / 2×) so they aren’t drowned out by long words.'],
          ['AdamW', 'The optimizer used; Adam with decoupled weight decay.'],
          ['Cosine LR with warmup', 'Learning-rate schedule: ramp up briefly, then decay along a cosine curve.'],
          ['Checkpoint', 'A saved model state (weights + config + step + val score).'],
        ]}
      />

      <h2>Reinforcement learning</h2>
      <DocTable
        headers={['Term', 'Meaning']}
        rows={[
          ['REINFORCE', 'The basic policy-gradient algorithm: play games, push up the probability of actions in winning episodes.'],
          ['PPO', 'Proximal Policy Optimization — clips each update so the policy can’t move too far at once.'],
          ['KL penalty', 'A term keeping the RL policy close to the supervised prior, preventing catastrophic forgetting.'],
          ['GRPO', 'Group-relative policy optimization — plays several games per word and uses their mean as a baseline.'],
          ['Advantage / baseline', 'How much better an action did than expected; the baseline is the expected return used to reduce variance.'],
        ]}
      />

      <h2>Evaluation &amp; deployment</h2>
      <DocTable
        headers={['Term', 'Meaning']}
        rows={[
          ['Win rate', 'Fraction of test games solved within six wrong guesses — the headline metric.'],
          ['Greedy decoding', 'Always guessing the highest-probability unguessed letter (no search or sampling).'],
          ['Held-out / OOL', 'Words in val/test that never appear in training (out-of-list) — so the model must generalise.'],
          ['Scaling law', 'The observed relationship between model size and win rate — logarithmic, peaking at d=768 here.'],
          ['Quantization (int8)', 'Compressing weights from 32-bit floats to 8-bit ints — ~4× smaller, used for the browser demo.'],
          ['ONNX', 'A portable model format; ONNX Runtime runs the model in the browser via WebAssembly.'],
        ]}
      />
    </DocLayout>
  )
}
