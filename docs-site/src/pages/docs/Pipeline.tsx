import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, CodeBlock, Diagram, Badge } from '../../framework'

export default function Pipeline() {
  return (
    <DocLayout>
      <h1>Training Pipeline &amp; Logs</h1>
      <div className="page-meta">
        <Badge type="python" label="methodology" />
        <Badge type="active" label="how it was built" />
      </div>

      <p>
        How a model goes from a word list to a 69% checkpoint: the training stages, what the logs
        look like and how to read them, and what ends up saved in a checkpoint. This documents the
        process — the <Link to="/docs/baselines">statistical baselines</Link> are the part you can
        run directly from this repo today.
      </p>

      <h2>The stages</h2>
      <Diagram>{`graph LR
    A["Word list<br/>317k train"] --> B["Supervised pretraining<br/>train_supervised.py"]
    B --> C["Best checkpoint<br/>supervised_best.pt"]
    C --> D["Evaluation<br/>evaluate.py · greedy"]
    C -.->|attempted| E["RL fine-tuning<br/>train_rl.py · no gain"]`}</Diagram>

      <DocTable
        headers={['Stage', 'Script', 'What it does']}
        rows={[
          ['Pretrain', 'train_supervised.py', 'Multi-label BCE on sampled board states'],
          ['Evaluate', 'evaluate.py', 'Greedy argmax on the 17,652-word test set'],
          ['RL (attempted)', 'train_rl.py', 'REINFORCE / KL-PPO / GRPO — none beat supervised'],
          ['Baselines', 'baselines/evaluate_ngram.py', 'Rule-based reference points (runnable here)'],
        ]}
      />

      <h2>Running a training search</h2>
      <p>
        Each architecture is launched with explicit flags; the search scripts wrap{' '}
        <code>train_supervised.py</code> with the right width, depth, learning rate, and curriculum.
      </p>

      <CodeBlock language="bash">{`# the d=768 best model (illustrative flags)
python scripts/train_supervised.py \\
  --train data/train_clean1.txt --val data/val_words.txt \\
  --d-model 768 --n-layers 6 --n-heads 12 --dim-ff 3072 \\
  --steps 200000 --lr 7e-5 \\
  --length-curriculum 4x2x \\
  --ckpt-dir logs/sup8_a_d768`}</CodeBlock>

      <h2>Reading the training logs</h2>
      <p>
        Supervised training logs two things: the BCE loss every step, and a greedy win rate on a
        2,000-word validation subset at each eval. The validation win rate — not the loss — is what
        progress is judged by.
      </p>

      <CodeBlock language="text" filename="supervised.log">{`[step 10000/200000]  loss=0.3905  lr=2.95e-04  elapsed=5m12s
[eval  step=10000]   val_win_rate=0.6810      <- 68.1% greedy on 2k val words
[step 20000/200000]  loss=0.3612  lr=2.81e-04  elapsed=10m24s
[eval  step=20000]   val_win_rate=0.6925
...
[best] val win-rate: 0.7052 -> saved supervised_best.pt`}</CodeBlock>

      <DocTable
        headers={['Field', 'Meaning']}
        rows={[
          ['loss', 'Multi-label BCE on the batch — should trend down, not the success metric'],
          ['val_win_rate', 'Greedy games won on 2k held-out words — the real progress signal'],
          ['lr', 'Current learning rate (cosine decay after warmup)'],
          ['best / saved', 'Checkpoint written only when validation win rate improves'],
        ]}
      />

      <Callout type="warning" title="Loss going down ≠ winning more">
        BCE loss and win rate correlate but diverge at the margin — a model can shave loss without
        winning more games. That’s why checkpoints are selected by validation win rate, and why the
        final number always comes from full-test greedy evaluation, never from training loss.
      </Callout>

      <h2>What’s inside a checkpoint</h2>
      <CodeBlock language="python">{`ckpt = torch.load("supervised_best.pt", weights_only=False)
ckpt.keys()
# -> ["model_state", "cfg", "step", "val_win_rate"]
#    model_state   : the weights
#    cfg           : architecture config (d_model, n_layers, ...)
#    step          : training step it was saved at
#    val_win_rate  : the validation score that earned the save`}</CodeBlock>

      <Callout type="note" title="No optimizer state">
        Checkpoints store only what’s needed to play: weights + config + provenance. There’s no
        optimizer state, so they’re small and load fast — which also made the int8 ONNX export for
        the <Link to="/play">browser demo</Link> straightforward.
      </Callout>

      <h2>Evaluating and reproducing the baselines</h2>
      <CodeBlock language="bash">{`# statistical baselines — runnable from this repo (needs numpy)
python3 -u baselines/evaluate_ngram.py            # both agents
python3 -u baselines/evaluate_ngram.py --agent ngram`}</CodeBlock>

      <p>
        For the results these pipelines produced, see{' '}
        <Link to="/docs/analysis">Performance Analysis</Link> and{' '}
        <Link to="/docs/experiments">Experiments &amp; Findings</Link>.
      </p>
    </DocLayout>
  )
}
