import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, Diagram, Badge } from '../../framework'

export default function Analysis() {
  return (
    <DocLayout>
      <h1>Performance Analysis</h1>
      <div className="page-meta">
        <Badge type="active" label="d=768 · 69.02%" />
        <Badge type="archived" label="17,652 test words" />
      </div>

      <p>
        With the model <Link to="/docs/training">trained</Link> and scaled to its best
        configuration (d=768, 43.2M params), the question becomes: how well does it actually play?
        This page breaks its performance on the held-out test set down by word length and by
        letter. The single number — 69.02% — hides two very different stories: the model is
        near-perfect on long words and intrinsically limited on short ones.
      </p>

      <Callout type="info" title="Headline numbers">
        <strong>69.02%</strong> win rate · 12,183 of 17,652 games won · greedy argmax decoding on
        words the model never saw in training.
      </Callout>

      <h2>What one game looks like</h2>
      <p>
        Before the aggregate numbers, here’s the model actually playing a real game — the secret
        word is <strong>basket</strong>. “Confidence” is how sure the model is of the letter it
        picks; watch it climb as the pattern fills in.
      </p>

      <DocTable
        headers={['Board', 'Guess', 'Confidence', 'Result']}
        rows={[
          ['_ _ _ _ _ _', 'e', '15%', '✓ hit'],
          ['_ _ _ _ e _', 'r', '16%', '✗ miss'],
          ['_ _ _ _ e _', 'd', '11%', '✗ miss'],
          ['_ _ _ _ e _', 's', '11%', '✓ hit'],
          ['_ _ s _ e _', 't', '21%', '✓ hit'],
          ['_ _ s _ e t', 'k', '30%', '✓ hit'],
          ['_ _ s k e t', 'a', '29%', '✓ hit'],
          ['_ a s k e t', 'b', '52%', '✓ solved → BASKET'],
        ]}
      />

      <Callout type="tip" title="What to notice">
        It opens with <code>e</code> — the most common English letter — then makes two reasonable
        misses (<code>r</code>, <code>d</code>) before the revealed letters start constraining
        things. Early guesses are low-confidence (~15%) because many letters are still plausible;
        as the board fills in, the model homes in fast — by the last blank it’s far more certain
        (<code>b</code> at 52%). Won in 8 guesses with 2 wrong, comfortably inside the 6-mistake
        budget. (Try your own words on the <Link to="/play">Play</Link> page.)
      </Callout>

      <h2>Win rate by word length</h2>
      <p>
        Performance climbs almost monotonically with length. Longer words expose more letters per
        correct guess and offer more positional structure to reason from; short words are
        information-starved.
      </p>

      <DocTable
        headers={['Length', 'Games', 'Win rate', 'Band']}
        rows={[
          ['4', '404', '12.6%', 'Short — very hard'],
          ['5', '857', '18.0%', 'Short — very hard'],
          ['6', '1,424', '32.0%', 'Short — hard'],
          ['7', '1,957', '49.3%', 'Medium'],
          ['8', '2,351', '60.8%', 'Medium'],
          ['9', '2,446', '74.2%', 'Medium — above average'],
          ['10–12', '5,544', '81–92%', 'Long — strong'],
          ['13–17', '2,531', '94–98%', 'Long — near perfect'],
          ['18+', '136', '~100%', 'Effectively solved'],
        ]}
      />

      <Diagram>{`graph LR
    A["len 4<br/>12.6%"] --> B["len 6<br/>32%"]
    B --> C["len 8<br/>61%"]
    C --> D["len 10<br/>81%"]
    D --> E["len 13<br/>94%"]
    E --> F["len 16+<br/>~98%"]`}</Diagram>

      <Callout type="warning" title="The short-word wall">
        A 4-letter word is won only 12.6% of the time. With six lives and so few positions, a
        single early miss is often fatal — and no model trained on this corpus broke through it.
        Two dedicated attempts (aggressive curriculum, a dedicated short-word model) both made it{' '}
        <em>worse</em>; see <Link to="/docs/experiments">Experiments</Link>.
      </Callout>

      <h2>Win rate by letter</h2>
      <p>
        For each letter, this is the win rate on games whose word contains that letter. Common
        letters are well above the 69% average; the three rarest — <code>j</code>, <code>k</code>,{' '}
        <code>w</code> — drag well below it.
      </p>

      <DocTable
        headers={['Tier', 'Letters', 'Win rate', 'Why']}
        rows={[
          ['Strongest', 't, p, c, i, n, r, s, l, e', '75–80%', 'Frequent — rich positional signal'],
          ['Around average', 'a, d, g, h, o, m, u, v, y', '71–75%', 'Common letters'],
          ['Below average', 'b, f, q, x, z', '61–67%', 'Less frequent'],
          ['Weak spots', 'w', '54.2%', 'Rare'],
          ['Weak spots', 'k', '44.9%', 'Rare'],
          ['Weak spots', 'j', '35.5%', 'Rarest — least signal'],
        ]}
      />

      <Callout type="note" title="Rare-letter weakness is intrinsic">
        <code>j/k/w</code> appear in few English words, so the model sees them rarely and they
        carry little positional information. Loss reweighting moved the overall score by only
        +0.12pp — this is a property of the language, not a fixable training bug. Words like{' '}
        <em>ajowan</em>, <em>alkyl</em>, and <em>aflow</em> are the typical losses.
      </Callout>

      <h2>The theoretical ceiling</h2>
      <p>
        69% is not far from the practical maximum for this setup. The test set contains a tail of
        words that are essentially unguessable from spelling statistics — rare abbreviations and
        foreign-derived strings (<em>xciv</em>, <em>vlbi</em>, <em>abfm</em>) — plus the
        information-starved short words above.
      </p>

      <DocTable
        headers={['', 'Win rate']}
        rows={[
          ['Best model (d=768)', '69.02%'],
          ['Estimated perfect play (informal)', '~75–80%'],
          ['Gap', 'Short words + unguessable tail'],
        ]}
      />

      <Callout type="tip" title="What would move the needle">
        Not a bigger model — d=1024 already regressed, and its validation also dropped (68.96% vs
        70.52% at d=768), suggesting under-training at the fixed budget rather than a proven data
        ceiling. Either way, more capacity stopped helping. The remaining gap is bounded by the
        corpus and by information theory. Real gains would require a much larger training
        vocabulary, or a fundamentally different prior (e.g. an LLM that already knows English
        morphology).
      </Callout>

      <p>
        For the full path that produced this model, see{' '}
        <Link to="/docs/experiments">Experiments &amp; Findings</Link>; to see it in action, head
        to <Link to="/play">Play</Link>.
      </p>
    </DocLayout>
  )
}
