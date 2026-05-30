import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, Badge, Diagram } from '../framework'

export default function HomePage() {
  return (
    <DocLayout>
      <h1>Hangman Transformer</h1>
      <div className="page-meta">
        <Badge type="active" label="69.02% win rate" />
        <Badge type="python" label="PyTorch" />
        <Badge type="js" label="ONNX · in-browser demo" />
      </div>

      <p>
        A character-level Transformer trained to play Hangman on words it has never seen.
        After 20+ experiments — supervised scaling, six reinforcement-learning attempts,
        curriculum tuning, inference-time search, and model specialisation — the best model
        reaches a <strong>69.02% win rate</strong> on 17,652 held-out test words.
      </p>

      <p>
        You can <Link to="/play">watch it play live in your browser</Link> — the model runs
        client-side via ONNX Runtime, no server involved.
      </p>

      <Callout type="tip" title="The short version">
        Supervised scaling was the only lever that reliably worked. Every other idea — RL,
        aggressive curriculum, candidate-pool search, a dedicated short-word model — either
        failed or regressed. The most valuable findings here are the <strong>negative</strong> ones,
        and <em>why</em> they failed.
      </Callout>

      <h2>The problem</h2>
      <p>
        Hangman is a sequential, partially-observable word game. A word is hidden; you see only
        its length as blanks and guess one letter at a time. Six wrong guesses and you lose.
        The model is evaluated on words it never saw in training — so it must learn the
        statistics of English spelling, not memorise a dictionary.
      </p>

      <p>
        For context, two non-neural <Link to="/docs/data-baselines">statistical baselines</Link>{' '}
        set the bar the model has to clear:
      </p>

      <DocTable
        headers={['', 'Win rate', 'What it is']}
        rows={[
          ['Statistical baseline — CandidateFrequency', '18.05%', 'Filter dictionary by pattern, guess by frequency'],
          ['Statistical baseline — NGramGlobal', '39.39%', 'Positional n-gram tables from 317k words'],
          ['→ Transformer (this project)', '69.02%', 'Best model — d=768, 43.2M params'],
          ['Estimated perfect play', '~75–80%', 'Some test words are unguessable abbreviations'],
        ]}
      />

      <p>
        The baselines, the data pipeline, and the runnable code are on the{' '}
        <Link to="/docs/data-baselines">Data &amp; Baselines</Link> page.
      </p>

      <h2>How it works</h2>
      <p>
        A pre-norm Transformer encoder reads the masked word plus the set of already-guessed
        letters, and outputs a probability for each of the 26 letters. At each turn it guesses
        the highest-scoring letter it hasn't tried yet.
      </p>

      <Diagram>{`graph LR
    A["Masked word<br/>_ r e _ _"] --> E[Transformer<br/>Encoder]
    G["Guessed letters<br/>{r, e, t}"] --> E
    E --> P["26 letter<br/>probabilities"]
    P --> M["Guess best<br/>unguessed letter"]`}</Diagram>

      <h2>The result that mattered: scaling</h2>
      <p>
        Model capacity (<code>d_model</code>) was the single reliable lever. Gains are
        logarithmic and peak at d=768 — beyond that, the 317k-word corpus can no longer support
        the parameters and performance regresses.
      </p>

      <DocTable
        headers={['Model', 'Params', 'Test win rate', 'Δ']}
        rows={[
          ['d=256', '3.25M', '66.20%', 'baseline'],
          ['d=384', '10.8M', '68.24%', '+2.04pp'],
          ['d=512', '19.2M', '68.79%', '+0.55pp'],
          ['d=768  ★ best', '43.2M', '69.02%', '+0.23pp'],
          ['d=1024', '76.8M', '67.69%', '−1.10pp — corpus too small'],
        ]}
      />

      <h2>What didn’t work (and why it’s interesting)</h2>
      <DocTable
        headers={['Attempt', 'Outcome', 'Root cause']}
        rows={[
          ['Reinforcement learning (×6)', 'Never beat supervised', 'Hangman is a knowledge task, not a strategy task'],
          ['Aggressive curriculum (16×/4×)', '−3pp regression', 'Starves long words to help short ones'],
          ['Inference-time candidate search', '−1 to −4pp', 'Test words are out-of-vocabulary; local stats mislead'],
          ['Dedicated short-word model', 'Worse on short words', 'Cross-length training transfers knowledge'],
          ['Scaling past d=768', '−1.10pp', 'Not enough data for the parameters'],
        ]}
      />

      <p>
        The full story, with numbers and reasoning for every run, is on the{' '}
        <Link to="/docs/experiments">Experiments &amp; Findings</Link> page.
      </p>

      <h2>Explore</h2>
      <DocTable
        headers={['Page', 'What’s there']}
        rows={[
          ['Play', 'Watch the model play, or play the same words yourself'],
          ['Experiments & Findings', 'Every run, the scaling law, and all 8 findings'],
        ]}
      />

      <Callout type="note" title="Reproducibility">
        Training code, data splits, and evaluation scripts live in the{' '}
        <a href="https://github.com/Mund99/hangman-transformer" target="_blank" rel="noreferrer">GitHub repo</a>.
        The in-browser demo uses an int8-quantised export of the d=512 model (18.8 MB,
        identical win rate to full precision).
      </Callout>
    </DocLayout>
  )
}
