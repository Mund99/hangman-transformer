import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, CodeBlock, Badge } from '../../framework'

export default function Baselines() {
  return (
    <DocLayout>
      <h1>Statistical Baselines</h1>
      <div className="page-meta">
        <Badge type="python" label="rule-based" />
        <Badge type="archived" label="reference only" />
      </div>

      <p>
        Before training anything neural, it helps to know how far a non-neural strategy gets.
        These two rule-based agents set the bar the Transformer has to clear. They are{' '}
        <strong>not the model</strong> — just reference points, evaluated on the same 17,652-word
        held-out <Link to="/docs/data">test set</Link>.
      </p>

      <DocTable
        headers={['Agent', 'Win rate', 'Avg wrong/game']}
        rows={[
          ['CandidateFrequency', '18.05%', '5.58'],
          ['NGramGlobal', '39.39%', '4.99'],
          ['Transformer (best)', '69.02%', '—'],
        ]}
      />

      <p>
        The code is in{' '}
        <a href="https://github.com/Mund99/hangman-transformer/blob/main/baselines/evaluate_ngram.py"
           target="_blank" rel="noreferrer"><code>baselines/evaluate_ngram.py</code></a>{' '}
        — runnable with just <code>numpy</code>.
      </p>

      <h2>1 · CandidateFrequency</h2>
      <p>
        Filter the training dictionary to words that still match the current masked pattern (and
        contain none of the wrong letters), then guess the most frequent letter among the
        survivors. The classic “narrow down the dictionary” strategy.
      </p>

      <CodeBlock language="python">{`def guess(self, masked, guessed, wrong):
    candidates = self.index.candidates(masked, wrong)   # still-consistent words
    if candidates:
        scores = Counter()
        for word in candidates:
            for pos, char in enumerate(word):
                if masked[pos] == "_" and char not in guessed:
                    scores[char] += 1
        if scores:
            return scores.most_common(1)[0][0]
    # no candidates left -> global letter frequency
    return next(l for l in self.fallback if l not in guessed)`}</CodeBlock>

      <p>
        It collapses on short words: a 4-letter word matches few dictionary entries, and a single
        wrong guess can wipe out the whole candidate set — after which it’s just blind frequency
        guessing.
      </p>

      <h2>2 · NGramGlobal</h2>
      <p>
        Build weighted positional n-gram tables (bigram through 9-gram) from the entire corpus.
        For each window of the masked word, every n-gram consistent with the known letters votes
        for the letters that could fill the blanks — weighted by corpus frequency and window
        length (<code>n³</code>, favouring longer, more specific context).
      </p>

      <CodeBlock language="python">{`for n in range(min(max_n, len(masked)), 1, -1):
    weight = n ** 3                      # longer grams carry more signal
    for window in sliding_windows(masked, n):
        grams = ngrams_matching_known_letters(window, n)
        for blank in window.blanks:
            for gram in grams:
                acc[gram[blank]] += gram.count * weight
return argmax(acc, excluding=guessed)`}</CodeBlock>

      <Callout type="tip" title="Why NGram more than doubles CandidateFrequency">
        Positional n-grams capture spelling structure — common prefixes, suffixes, and letter
        clusters — that survives even when no full dictionary word matches. That’s worth +21pp.
        But it still scores each window independently with no view of the whole word, which is
        exactly the gap the Transformer closes.
      </Callout>

      <h2>Where both break: word length</h2>
      <p>
        Both agents are sharply length-dependent. Long words offer plenty of structural
        constraints; short words offer almost none.
      </p>

      <DocTable
        headers={['Length', 'Games', 'CandidateFreq', 'NGramGlobal', 'Assessment']}
        rows={[
          ['4–6', '2,685', '4–9%', '12–18%', 'Both fail — too few constraints'],
          ['7–9', '6,754', '13–16%', '22–33%', 'Weak'],
          ['10–12', '5,544', '19–23%', '40–58%', 'NGram improving'],
          ['13+', '2,672', '27–63%', '67–95%', 'NGram strong on long words'],
        ]}
      />

      <Callout type="warning" title="The short-word problem starts here">
        Neither baseline cracks short words — and, as the{' '}
        <Link to="/docs/experiments">experiments</Link> show, neither does the neural model,
        despite several dedicated attempts. Short words are information-starved by nature.
      </Callout>

      <h2>Run them yourself</h2>
      <CodeBlock language="bash">{`# from the repo root (needs numpy)
python3 -u baselines/evaluate_ngram.py            # both agents
python3 -u baselines/evaluate_ngram.py --agent freq
python3 -u baselines/evaluate_ngram.py --agent ngram`}</CodeBlock>

      <p>
        Next: how the <Link to="/docs/architecture">Transformer</Link> is built to beat these by
        30–50 points.
      </p>
    </DocLayout>
  )
}
