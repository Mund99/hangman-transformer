import { Link } from 'react-router-dom'
import { DocLayout, DocTable, Callout, CodeBlock, Diagram, Badge } from '../../framework'

export default function Architecture() {
  return (
    <DocLayout>
      <h1>Architecture</h1>
      <div className="page-meta">
        <Badge type="python" label="PyTorch" />
        <Badge type="active" label="HangmanPolicy" />
      </div>

      <p>
        The model is a pre-norm Transformer <strong>encoder</strong> that reads the whole board at
        once and outputs a probability for each of the 26 letters. The architecture is fixed
        across every experiment — only its width (<code>d_model</code>), depth, and head count
        change. The class is called <code>HangmanPolicy</code>.
      </p>

      <Callout type="note" title="Why an encoder (not a decoder)">
        A decoder generates text left-to-right, one token at a time. Here there’s nothing to
        generate — the entire board is visible up front, and we just need to score 26 letters. An
        encoder reads the whole pattern at once and lets every position inform every other, which is
        exactly what’s needed to reason about which letters fit the gaps.
      </Callout>

      <h2>The forward pass at a glance</h2>
      <Diagram>{`graph TD
    T["token_ids · 45<br/>masked word"] --> TE["Token<br/>embedding"]
    POS["positions · 45"] --> PE["Position<br/>embedding"]
    G["guessed_vec · 26"] --> GP["Linear<br/>26 to d"]
    TE --> SUM["sum"]
    PE --> SUM
    GP --> SUM
    SUM --> ENC["Transformer encoder<br/>pre-norm, GELU"]
    ENC --> POOL["Masked mean pool<br/>45×d to d"]
    POOL --> MLP["MLP head<br/>d to d to 26"]
    MLP --> OUT["26 letter logits"]`}</Diagram>

      <h2>Inputs: three streams, summed</h2>
      <p>
        The board reaches the model as three tensors (built on the{' '}
        <Link to="/docs/data">Data</Link> page), which are projected to the model width and added
        together so every position carries the full game context.
      </p>

      <DocTable
        headers={['Stream', 'From', 'Encodes']}
        rows={[
          ['Token embedding', 'token_ids [45]', 'Which letter (or blank/pad) sits at each position'],
          ['Position embedding', 'positions [45]', 'Where in the word each token is'],
          ['Guessed projection', 'guessed_vec [26]', 'Which letters have already been tried (global context)'],
        ]}
      />

      <CodeBlock language="python" filename="model.py (forward)">{`x = self.tok_embed(token_ids) + self.pos_embed(positions)   # [B, 45, d]
g = self.guessed_proj(guessed_vec).unsqueeze(1)             # [B, 1, d]
x = x + g                                                   # broadcast to all positions
h = self.encoder(x, src_key_padding_mask=~attn_mask.bool()) # [B, 45, d]`}</CodeBlock>

      <Callout type="tip" title="Why broadcast the guessed letters to every position">
        The set of already-guessed letters is global information — it applies equally everywhere.
        Adding it to every position means each character’s representation is immediately aware of
        the full game state, without the model having to learn to route that information itself.
      </Callout>

      <h2>The encoder</h2>
      <p>
        A stack of standard Transformer encoder layers — multi-head self-attention plus a
        feed-forward network, each wrapped in a residual connection with{' '}
        <strong>pre-norm</strong> (LayerNorm inside the residual branch) and{' '}
        <strong>GELU</strong> activations. Pre-norm keeps gradients stable as the model gets wider
        and deeper.
      </p>

      <Callout type="tip" title="What self-attention does here, in plain terms">
        For the board <code>_ r e _ _</code>, the first blank doesn’t decide in isolation — it{' '}
        <em>looks at</em> the other positions and weighs how much each one constrains it. Seeing{' '}
        <code>r</code> and <code>e</code> next to it makes letters like <code>p</code> (“pre…”) or{' '}
        <code>t</code> (“tre…”) far more likely than, say, <code>q</code>. Self-attention is exactly
        this: every position gathers information from every other and updates its own
        representation. “Multi-head” just runs several of these comparisons in parallel, each free to
        focus on a different kind of relationship (adjacent letters, vowel/consonant patterns, word
        endings).
      </Callout>

      <CodeBlock language="python">{`# pre-norm block (norm_first=True)
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))`}</CodeBlock>

      <h2>Pooling and head</h2>
      <p>
        After the encoder, the per-position vectors are collapsed into one word-level vector by{' '}
        <strong>masked mean pooling</strong> (averaging only the real, non-padding positions). A
        small two-layer MLP then maps that to 26 logits — one per letter.
      </p>

      <CodeBlock language="python">{`mask   = attn_mask.unsqueeze(-1).float()
pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # [B, d]
logits = self.head(pooled)                              # [B, 26]`}</CodeBlock>

      <h3>Making a guess — <code>predict_letter()</code></h3>
      <p>
        At play time the model runs a forward pass, masks out every letter already guessed by
        setting its logit to −∞, and returns the argmax of what remains. Plain greedy decoding,
        wrapped in <code>@torch.no_grad()</code>.
      </p>

      <CodeBlock language="python">{`@torch.no_grad()
def predict_letter(self, token_ids, attn_mask, guessed_vec):
    logits = self.forward(token_ids, attn_mask, guessed_vec)  # [1, 26]
    logits = logits.masked_fill(guessed_vec.bool(), float("-inf"))
    return chr(int(logits.argmax()) + 97)                     # 'a'..'z'`}</CodeBlock>

      <Callout type="note" title="Why no beam search or sampling">
        Every attempt to be cleverer at inference time — beam search, sampling, candidate-pool
        re-ranking — made it <em>worse</em> (see <Link to="/docs/experiments">Experiments</Link>).
        The model’s own probability ranking is already the right ordering, so greedy argmax is
        both simplest and best.
      </Callout>

      <h2>Where the parameters live</h2>
      <p>
        Almost all of the model’s weight is in the Transformer layers; the embeddings and head are
        tiny by comparison. The attention and feed-forward projections scale with{' '}
        <code>d_model²</code>, which is what drives the parameter count up as the model widens.
      </p>

      <DocTable
        headers={['Component', 'Rough share', 'Scales with']}
        rows={[
          ['Transformer encoder layers', '~95%', 'd_model² × n_layers'],
          ['Token + position embeddings', '~2%', 'vocab × d_model'],
          ['Guessed-letter projection', '<1%', '26 × d_model'],
          ['MLP head', '~2%', 'd_model²'],
        ]}
      />

      <p>
        The 26 logits are treated as 26 <em>independent</em> yes/no predictions (a word has several
        letters at once), so training uses multi-label binary cross-entropy rather than a softmax —
        the full details are on the <Link to="/docs/training">Training</Link> page.
      </p>

      <h2>Parameter count &amp; the scaling law</h2>
      <p>
        Capacity is the one lever that reliably improved results. Each doubling of width bought
        roughly half the previous gain — a clean logarithmic curve that <em>peaks at d=768</em>{' '}
        and then reverses, because the 317k-word corpus can’t support more parameters.
      </p>

      <DocTable
        headers={['d_model', 'Layers', 'Heads', 'Params', 'Test %', 'Δ']}
        rows={[
          ['256', '4', '4', '3.25M', '66.20%', 'baseline'],
          ['384', '6', '6', '10.8M', '68.24%', '+2.04pp'],
          ['512', '6', '8', '19.2M', '68.79%', '+0.55pp'],
          ['768  ★', '6', '12', '43.2M', '69.02%', '+0.23pp'],
          ['1024', '6', '16', '76.8M', '67.69%', '−1.10pp'],
        ]}
      />

      <Diagram>{`graph LR
    A["d=256<br/>66.2%"] --> B["d=384<br/>68.2%"]
    B --> C["d=512<br/>68.8%"]
    C --> D["d=768 ★<br/>69.0%"]
    D --> E["d=1024<br/>67.7% ▼"]`}</Diagram>

      <Callout type="warning" title="The corpus is the ceiling, not the architecture">
        At d=1024 (76.8M params) the model regressed 1.10pp. Beyond d=768 there simply isn’t
        enough training signal per parameter. Pushing past this point would need a much larger
        corpus, not a bigger model.
      </Callout>

      <p>
        How this architecture is actually trained — and why reinforcement learning couldn’t
        improve it — is on the <Link to="/docs/training">Training</Link> page.
      </p>
    </DocLayout>
  )
}
