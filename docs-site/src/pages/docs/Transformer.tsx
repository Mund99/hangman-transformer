import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, CodeBlock, Diagram, Badge } from '../../framework'

export default function Transformer() {
  return (
    <DocLayout>
      <h1>Transformer Explained</h1>
      <div className="page-meta">
        <Badge type="active" label="primer" />
        <Badge type="js" label="no math required" />
      </div>

      <p>
        A short, intuition-first primer on the architecture this whole project is built on. If you
        already know how Transformers work, skip to <Link to="/docs/architecture">Architecture</Link>{' '}
        to see the hangman-specific design. If not, this page gives you just enough to follow along —
        examples are tied to a real hangman board rather than abstract math.
      </p>

      <h2>The problem it solves</h2>
      <p>
        To guess a letter in <code>_ r e _ _</code>, you need to look at <em>all</em> the revealed
        letters at once — the <code>r</code> and the <code>e</code> together suggest very different
        words than either alone. The challenge for a neural network is: how does each position
        “see” all the others and combine that information?
      </p>
      <p>
        Older sequence models (RNNs/LSTMs) read left-to-right, one position at a time, squeezing
        everything seen so far into a single running memory. That memory is a bottleneck — distant
        positions fade, and the strictly sequential processing can’t be parallelised. The
        Transformer throws that out.
      </p>

      <h2>The core idea: self-attention</h2>
      <p>
        Self-attention lets <strong>every position look directly at every other position</strong>,
        in one step, and decide how much each one matters. No running memory, no left-to-right
        bottleneck — the blank in <code>_ r e _ _</code> can attend straight to the <code>r</code>{' '}
        and <code>e</code> and weigh them however it likes.
      </p>

      <Diagram>{`graph TD
    B1["blank #1"] --> A["self-attention<br/>who matters to me?"]
    R["r"] --> A
    E["e"] --> A
    B2["blank #4"] --> A
    B3["blank #5"] --> A
    A --> O["updated blank #1<br/>informed by r, e"]`}</Diagram>

      <h2>Query, Key, Value — the intuition</h2>
      <p>
        Attention is often explained with a library analogy. Each position produces three things:
      </p>
      <DocTable
        headers={['Name', 'Think of it as', 'For the blank in “_ r e _ _”']}
        rows={[
          ['Query (Q)', '“what am I looking for?”', '“I’m an empty slot — what nearby letters constrain me?”'],
          ['Key (K)', '“what do I offer?”', 'The r and e each advertise “I’m a consonant/vowel here”'],
          ['Value (V)', '“the info I’ll hand over”', 'The actual content passed along once matched'],
        ]}
      />
      <p>
        Each position’s <strong>query</strong> is compared against every other position’s{' '}
        <strong>key</strong>; the better the match, the more of that position’s{' '}
        <strong>value</strong> gets mixed in. Stronger matches → bigger influence. That’s the whole
        mechanism.
      </p>
      <CodeBlock language="text">{`attention(Q, K, V) = softmax( Q · Kᵀ / √d ) · V
                     └─── match scores ───┘   └ blended info`}</CodeBlock>

      <Callout type="tip" title="Why it’s a good fit for hangman">
        A hangman board is a fixed set of positions where every blank needs to reason about every
        revealed letter at once. That’s exactly what self-attention does — which is why a
        Transformer beats the older n-gram approaches that only look at small fixed windows.
      </Callout>

      <h2>Multi-head attention</h2>
      <p>
        One attention pass captures one kind of relationship. <strong>Multi-head</strong> attention
        runs several in parallel, each free to focus on something different — one head might track
        adjacent letters, another vowel/consonant rhythm, another common word-endings. Their
        results are concatenated and combined.
      </p>

      <h2>The Transformer block</h2>
      <p>
        A Transformer layer wraps two operations, each with a residual connection (add the input
        back) and a normalisation step that keeps training stable:
      </p>
      <CodeBlock language="text">{`x = x + SelfAttention(LayerNorm(x))   # mix information across positions
x = x + FeedForward(LayerNorm(x))     # process each position on its own`}</CodeBlock>
      <p>
        <strong>Self-attention</strong> moves information <em>between</em> positions;{' '}
        the <strong>feed-forward network</strong> then transforms each position on its own. Stack a
        few of these blocks and the representation gets progressively richer. (Our model stacks 4–6.)
      </p>

      <h2>Encoder vs decoder</h2>
      <p>
        Transformers come in two flavours, and the difference is just what each position is allowed
        to attend to:
      </p>
      <DocTable
        headers={['Type', 'Each position sees', 'Used for', 'Examples']}
        rows={[
          ['Encoder', 'Every position (bidirectional)', 'Understanding a fixed input', 'BERT, our hangman model'],
          ['Decoder', 'Only earlier positions (causal)', 'Generating text left-to-right', 'GPT, Llama'],
        ]}
      />

      <Callout type="note" title="Hangman uses an encoder">
        There’s nothing to generate — the whole board is visible up front and we just need to score
        26 letters. So every position should see every other (bidirectional), which is exactly what
        an encoder does. No left-to-right masking required.
      </Callout>

      <h2>That’s enough to continue</h2>
      <p>
        You now have the mental model: self-attention lets every position weigh every other, multi-head
        runs several such comparisons in parallel, and stacking blocks builds up a rich
        representation. See how this is wired into the actual model — three input streams, pooling,
        and the 26-letter head — on the <Link to="/docs/architecture">Architecture</Link> page.
      </p>
    </DocLayout>
  )
}
