import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, CodeBlock, Diagram, Badge } from '../../framework'

export default function Data() {
  return (
    <DocLayout>
      <h1>The Problem &amp; Data</h1>
      <div className="page-meta">
        <Badge type="active" label="317k words" />
        <Badge type="python" label="data pipeline" />
      </div>

      <h2>The game</h2>
      <p>
        Hangman is a sequential, partially-observable word game. A word is chosen from a
        dictionary; the player sees only its length as blanks and guesses one letter at a time.
        A correct guess reveals every position of that letter; a wrong guess costs a life. After
        six wrong guesses, the game is lost.
      </p>

      <Diagram>{`graph LR
    A["_ _ _ _ _ _"] -->|guess e| B["_ _ _ _ e _"]
    B -->|guess r| C["_ _ _ _ e r"]
    C -->|guess a ✗| D["_ _ _ _ e r<br/>1 wrong"]
    D -->|guess o| E["_ o _ _ e r"]
    E -->|...| F["solved or lost"]`}</Diagram>

      <Callout type="info" title="Why it’s a real machine-learning problem">
        The test words are <strong>disjoint</strong> from the training words. You cannot memorise
        the answers — winning requires learning the statistics of English spelling: which letters
        tend to appear given a partial pattern. It is a knowledge problem first, a strategy
        problem second.
      </Callout>

      <h2>The corpus</h2>
      <p>
        <strong>353,046</strong> unique English words, sourced from NLTK (WordNet lemmas + the{' '}
        <code>words</code> corpus) and a competition word list, then deduplicated, lowercased,
        filtered to alphabetic words of <strong>length ≥ 4</strong>, and split <strong>90 / 5 / 5</strong>{' '}
        into disjoint train / validation / test sets.
      </p>

      <DocTable
        headers={['Split', 'File', 'Words', 'Share', 'Role']}
        rows={[
          ['Train', 'train_clean1.txt', '317,742', '90%', 'Learning the model'],
          ['Validation', 'val_words.txt', '17,652', '5%', 'Checkpoint selection during training'],
          ['Test', 'test_words.txt', '17,652', '5%', 'Final evaluation — never seen in training'],
          ['(subset)', 'train_short46.txt', '48,906', '—', 'Length 4–6 words (short-word experiment)'],
        ]}
      />

      <Callout type="info" title="Why a held-out test set matters here">
        The three splits share no words. Validation guides <em>which checkpoint to keep</em>; the
        test set is touched only once, for the final number. Because the model is graded on words
        it has never seen, it cannot memorise answers — it must generalise the statistics of
        English spelling.
      </Callout>

      <Callout type="note" title="Data cleaning barely mattered">
        A later cleaning pass removed only 248 of ~318k training words (&lt; 0.1%) and moved
        results by &lt; 0.1pp. Corpus quality was never the bottleneck — model capacity and the
        intrinsic difficulty of short words were.
      </Callout>

      <h2>Turning a word into training examples</h2>
      <p>
        A single word like <code>apple</code> can appear in many different game states — fully
        blank, half-revealed, nearly solved. Instead of always starting blank,{' '}
        <code>sample_state()</code> draws a random mid-game snapshot each time a word is used, so
        the model learns every phase of a game rather than just the opening move.
      </p>

      <CodeBlock language="python" filename="hangman/dataset.py (simplified)">{`def sample_state(word, rng):
    unique = list(set(word))
    rng.shuffle(unique)

    # k correctly-revealed letters — uniform over [0, |unique|-1]
    k = rng.randint(0, max(len(unique) - 1, 0))
    revealed = set(unique[:k])

    # j wrong guesses already made — Geometric(0.35), capped at 5
    not_in_word = [c for c in LETTERS if c not in set(word)]
    rng.shuffle(not_in_word)
    j = min(5, geometric(rng, 0.35))
    wrong = set(not_in_word[:j])

    masked  = "".join(c if c in revealed else "_" for c in word)
    guessed = revealed | wrong          # both reveal correct + wrong guesses
    return masked, guessed`}</CodeBlock>

      <Callout type="tip" title="Worked example — word = “apple”">
        Say the sampler picks <code>k = 2</code> revealed letters{' '}
        <code>{'{a, l}'}</code> and <code>j = 1</code> wrong guess <code>{'{s}'}</code>:
        <ul>
          <li><strong>masked</strong> → <code>a _ _ l _</code></li>
          <li><strong>guessed</strong> → <code>{'{a, l, s}'}</code> (the board state the model sees)</li>
          <li><strong>target</strong> → letters actually in the word: <code>{'{a, p, l, e}'}</code></li>
          <li><strong>loss mask</strong> → only score letters not yet guessed — so the model is
            graded on whether it predicts <code>p</code> and <code>e</code> (and correctly rejects
            the other unguessed letters)</li>
        </ul>
        Both <code>k</code> and <code>j</code> are re-sampled every epoch, so <code>apple</code>{' '}
        becomes a fresh training example each time it appears.
      </Callout>

      <h2>How the board is encoded</h2>
      <p>
        Each game state becomes three tensors the model can read. (The full architecture is on the{' '}
        <Link to="/docs/architecture">Architecture</Link> page; this is the input side.)
      </p>

      <DocTable
        headers={['Input', 'Shape', 'Meaning']}
        rows={[
          ['token_ids', '[45]', 'Masked word: 0=pad, 1=blank, 2–27 = a–z'],
          ['attn_mask', '[45]', '1 for real positions, 0 for padding'],
          ['guessed_vec', '[26]', 'Binary — which letters have been tried'],
        ]}
      />

      <CodeBlock language="python">{`# word = "apple", guessed = {a, l, s}  ->  masked = "a _ _ l _"
token_ids = [2, 1, 1, 13, 1, 0, 0, ...]   # a=2, blank=1, l=13, pad=0
guessed   = [1,0,0,...,1,...,1,...,0]      # a, l, s flagged`}</CodeBlock>

      <Callout type="tip" title="Two distinct tokens for blank vs pad">
        A blank (<code>_</code>, token 1) means “a letter exists here but is hidden”; padding
        (token 0) means “there is no letter here at all.” Keeping them separate lets the model use
        word length as a signal.
      </Callout>

      <p>
        Next: the <Link to="/docs/baselines">statistical baselines</Link> that set the bar, or the{' '}
        <Link to="/docs/architecture">model architecture</Link>.
      </p>
    </DocLayout>
  )
}
