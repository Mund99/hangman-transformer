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
        After 20+ experiments — supervised scaling, reinforcement-learning attempts,
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
        For context, two non-neural <Link to="/docs/baselines">statistical baselines</Link>{' '}
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
        See the <Link to="/docs/data">data pipeline</Link> and the{' '}
        <Link to="/docs/baselines">statistical baselines</Link> (with runnable code) for the full
        picture.
      </p>

      <h2>How it works</h2>
      <p>
        A pre-norm Transformer encoder reads the masked word plus the set of already-guessed
        letters, and outputs a probability for each of the 26 letters. At each turn it guesses
        the highest-scoring letter it hasn't tried yet.
      </p>

      <Diagram>{`graph LR
    A["Masked word<br/>_ r e _ _"] --> E["Transformer<br/>Encoder"]
    G["Guessed letters<br/>r, e, t"] --> E
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
          ['d=768 ★ best', '43.2M', '69.02%', '+0.23pp'],
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

      <h2>How to read these docs</h2>
      <p>The pages follow the project end to end — feel free to jump to what interests you.</p>
      <DocTable
        headers={['Section', 'Page', 'What’s there']}
        rows={[
          ['Foundations', 'The Problem & Data', 'Game rules, the 90/5/5 corpus, how a board becomes training data'],
          ['Foundations', 'Statistical Baselines', 'Two rule-based agents (18% & 39%) that set the bar'],
          ['The Model', 'Transformer Explained', 'A plain-language primer on self-attention (no math)'],
          ['The Model', 'Architecture', 'The Transformer, encoding, inference, and the scaling law'],
          ['The Model', 'Training: Supervised & RL', 'How it’s trained — and exactly why 6 RL attempts failed'],
          ['Results', 'Performance Analysis', 'The best model by word length and by letter'],
          ['Results', 'Experiments & Findings', 'Every run in order, with reasoning, plus 8 findings'],
          ['Reference', 'Training Pipeline & Logs', 'Stages, log format, checkpoint structure'],
          ['Reference', 'Glossary', 'Definitions for every term used'],
          ['Try it', 'Play', 'Watch the model play, or play the same words yourself'],
        ]}
      />

      <Callout type="tip" title="New to this? Start here">
        For the gentlest path, read in this order:{' '}
        <Link to="/docs/transformer">Transformer Explained</Link> (no-math primer) →{' '}
        <Link to="/docs/data">The Problem &amp; Data</Link> (what the game and data are) →{' '}
        <Link to="/play">Play</Link> (watch it run). Then come back for the rest.
      </Callout>

      <Callout type="note" title="Already know ML? The quick tour">
        The <Link to="/docs/experiments">Experiments &amp; Findings</Link> page plus{' '}
        <Link to="/play">Play</Link> tell the whole story fast: what worked, what didn’t, and how
        it feels.
      </Callout>

      <Callout type="note" title="What’s in the repo">
        The{' '}
        <a href="https://github.com/Mund99/hangman-transformer" target="_blank" rel="noreferrer">GitHub repo</a>{' '}
        contains the full training code (<code>hangman/</code>, <code>scripts/</code>), every
        experiment’s run config, the data splits, runnable statistical baselines, and curated
        result logs. The in-browser demo runs an int8-quantised <strong>d=512</strong> model
        (18.8 MB) for a fast download — the quantization barely affects accuracy — within ~0.2pp of the
        best d=768 model.
      </Callout>
    </DocLayout>
  )
}
