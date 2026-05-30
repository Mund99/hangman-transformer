import { useState, useEffect, useRef, useCallback } from 'react'
import * as ort from 'onnxruntime-web'
import './Play.css'

// Load ORT's WASM binaries from the matching CDN version — robust in production builds.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'

// Asset base (respects Vite base path on GitHub Pages)
const BASE = import.meta.env.BASE_URL

// ── Vocab / encoding ────────────────────────────────────────────────────────
const MAX_LEN = 45
const charToToken = (c: string) => (c === '_' ? 1 : c.charCodeAt(0) - 97 + 2)

function encode(masked: string, guessed: Set<string>) {
  const tokenIds = new BigInt64Array(MAX_LEN)
  const attnMask = new BigInt64Array(MAX_LEN)
  const guessedVec = new Float32Array(26)
  for (let i = 0; i < masked.length; i++) {
    tokenIds[i] = BigInt(charToToken(masked[i]))
    attnMask[i] = 1n
  }
  for (const c of guessed) guessedVec[c.charCodeAt(0) - 97] = 1.0
  return {
    token_ids:   new ort.Tensor('int64',   tokenIds,   [1, MAX_LEN]),
    attn_mask:   new ort.Tensor('int64',   attnMask,   [1, MAX_LEN]),
    guessed_vec: new ort.Tensor('float32', guessedVec, [1, 26]),
  }
}

async function getProbs(session: ort.InferenceSession, masked: string, guessed: Set<string>) {
  const result = await session.run(encode(masked, guessed))
  const logits = Array.from(result.logits.data as Float32Array)
  const maxL = Math.max(...logits)
  const exps = logits.map(v => Math.exp(v - maxL))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(v => v / sum)
  for (const c of guessed) probs[c.charCodeAt(0) - 97] = 0
  const total = probs.reduce((a, b) => a + b, 0)
  return total > 0 ? probs.map(v => v / total) : probs
}

// ── Game logic ──────────────────────────────────────────────────────────────
type Mode = 'ai' | 'user'
type Status = 'playing' | 'won' | 'lost'
interface Guess { letter: string; hit: boolean }
interface GameState {
  word: string; masked: string; guessed: Set<string>
  wrong: number; history: Guess[]; status: Status
}

const initGame = (word: string): GameState => ({
  word, masked: '_'.repeat(word.length), guessed: new Set(), wrong: 0, history: [], status: 'playing',
})

function applyGuess(g: GameState, letter: string): GameState {
  const guessed = new Set(g.guessed); guessed.add(letter)
  const hit = g.word.includes(letter)
  const wrong = hit ? g.wrong : g.wrong + 1
  const masked = [...g.word].map(c => (guessed.has(c) ? c : '_')).join('')
  const history = [...g.history, { letter, hit }]
  let status: Status = 'playing'
  if (!masked.includes('_')) status = 'won'
  else if (wrong >= 6) status = 'lost'
  return { ...g, masked, guessed, wrong, history, status }
}

const LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('')
const KBD_ROWS = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']

// ── Gallows drawing ─────────────────────────────────────────────────────────
function Gallows({ wrong }: { wrong: number }) {
  const limbs = [
    { d: 'circle', cx: 62, cy: 40, r: 13 },              // head
    { d: 'line', x1: 62, y1: 53, x2: 62, y2: 92 },       // body
    { d: 'line', x1: 62, y1: 62, x2: 42, y2: 80 },       // left arm
    { d: 'line', x1: 62, y1: 62, x2: 82, y2: 80 },       // right arm
    { d: 'line', x1: 62, y1: 92, x2: 44, y2: 118 },      // left leg
    { d: 'line', x1: 62, y1: 92, x2: 80, y2: 118 },      // right leg
  ]
  return (
    <svg width="128" height="140" viewBox="0 0 128 140" fill="none">
      {/* gallows frame */}
      <g stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity="0.85">
        <line x1="14" y1="134" x2="92" y2="134" />
        <line x1="30" y1="134" x2="30" y2="12" />
        <line x1="30" y1="12" x2="62" y2="12" />
        <line x1="62" y1="12" x2="62" y2="26" strokeWidth="2" />
      </g>
      {/* body parts */}
      <g stroke="var(--miss)" strokeWidth="2.5" strokeLinecap="round" fill="none">
        {limbs.slice(0, wrong).map((l, i) =>
          l.d === 'circle'
            ? <circle key={i} className="limb" cx={l.cx} cy={l.cy} r={l.r} />
            : <line key={i} className="limb" x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} />
        )}
      </g>
    </svg>
  )
}

// ── Icons ───────────────────────────────────────────────────────────────────
const IconBrain = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
  </svg>
)
const IconUser = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
  </svg>
)

// ── Main ────────────────────────────────────────────────────────────────────
export default function Play() {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [progress, setProgress] = useState(0)
  const [words, setWords] = useState<string[]>([])
  const [mode, setMode] = useState<Mode>('ai')
  const [game, setGame] = useState<GameState | null>(null)
  const [probs, setProbs] = useState<number[]>(new Array(26).fill(0))
  const [next, setNext] = useState<string | null>(null)
  const [auto, setAuto] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [showHint, setShowHint] = useState(false)
  const [custom, setCustom] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  const sessRef = useRef<ort.InferenceSession | null>(null)
  const gameRef = useRef<GameState | null>(null)

  useEffect(() => { gameRef.current = game }, [game])

  // ── Load model (with download progress) + words ──
  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const resp = await fetch(`${BASE}hangman.onnx`)
        const total = Number(resp.headers.get('content-length')) || 19_000_000
        const reader = resp.body!.getReader()
        const chunks: Uint8Array[] = []
        let received = 0
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value)
          received += value.length
          if (!cancelled) setProgress(Math.min(99, Math.round((received / total) * 100)))
        }
        const buf = new Uint8Array(received)
        let off = 0
        for (const c of chunks) { buf.set(c, off); off += c.length }
        const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] })
        const txt = await fetch(`${BASE}words.txt`).then(r => r.text())
        if (cancelled) return
        sessRef.current = session
        setWords(txt.trim().split('\n').filter(Boolean))
        setStatus('ready')
      } catch {
        if (!cancelled) setStatus('error')
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  const refreshProbs = useCallback(async (g: GameState) => {
    if (!sessRef.current || g.status !== 'playing') { setNext(null); return }
    const p = await getProbs(sessRef.current, g.masked, g.guessed)
    setProbs(p)
    setNext(String.fromCharCode(97 + p.indexOf(Math.max(...p))))
  }, [])

  const start = useCallback(async (word: string) => {
    const g = initGame(word)
    setGame(g); gameRef.current = g
    setShowCustom(false)
    setAuto(false)
    setProbs(new Array(26).fill(0)); setNext(null)
    await refreshProbs(g)
  }, [refreshProbs])

  const startRandom = useCallback(() => {
    if (!words.length) return
    start(words[Math.floor(Math.random() * words.length)])
  }, [words, start])

  const startCustom = useCallback(() => {
    const w = custom.toLowerCase().trim()
    if (!/^[a-z]{2,50}$/.test(w)) return
    setCustom('')
    start(w)
  }, [custom, start])

  // auto-start first game
  useEffect(() => {
    if (status === 'ready' && words.length && !game) startRandom()
  }, [status, words, game, startRandom])

  // ── AI step ──
  const step = useCallback(async () => {
    const g = gameRef.current
    if (!sessRef.current || !g || g.status !== 'playing') return
    setThinking(true)
    const p = await getProbs(sessRef.current, g.masked, g.guessed)
    const letter = String.fromCharCode(97 + p.indexOf(Math.max(...p)))
    const ng = applyGuess(g, letter)
    setGame(ng); gameRef.current = ng
    setThinking(false)
    await refreshProbs(ng)
  }, [refreshProbs])

  // ── User guess ──
  const guess = useCallback(async (letter: string) => {
    const g = gameRef.current
    if (!g || g.status !== 'playing' || g.guessed.has(letter)) return
    const ng = applyGuess(g, letter)
    setGame(ng); gameRef.current = ng
    await refreshProbs(ng)
  }, [refreshProbs])

  // physical keyboard in user mode
  useEffect(() => {
    if (mode !== 'user') return
    const h = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase()
      if (/^[a-z]$/.test(k)) guess(k)
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [mode, guess])

  // auto-play loop
  useEffect(() => {
    if (mode !== 'ai' || !auto || !game || game.status !== 'playing') {
      if (game && game.status !== 'playing') setAuto(false)
      return
    }
    const t = setTimeout(() => step(), 620)
    return () => clearTimeout(t)
  }, [mode, auto, game, step])

  const switchMode = (m: Mode) => {
    setMode(m)
    setAuto(false)
    if (game) start(game.word)  // restart same word in new mode
  }

  // ── Render helpers ──
  const sortedLetters = game
    ? [...LETTERS].sort((a, b) => {
        const ga = game.guessed.has(a), gb = game.guessed.has(b)
        if (ga !== gb) return ga ? 1 : -1
        return probs[b.charCodeAt(0) - 97] - probs[a.charCodeAt(0) - 97]
      })
    : LETTERS

  const aiHintActive = mode === 'user' && showHint && game?.status === 'playing'

  return (
    <div className="main">
      <div className="page hg-page">
        <div className="hg">

          {/* ── Header ── */}
          <div className="hg-head">
            <h1 className="hg-title"><span className="dot" />Hangman</h1>
            <p className="hg-sub">
              Watch the model guess a word, or play the same words yourself.
            </p>

            <div className="hg-modes" role="tablist">
              <button className="hg-mode" data-on={mode === 'ai'} onClick={() => switchMode('ai')}>
                <IconBrain /> Watch AI
              </button>
              <button className="hg-mode" data-on={mode === 'user'} onClick={() => switchMode('user')}>
                <IconUser /> Play Yourself
              </button>
            </div>
          </div>

          {/* ── Loading ── */}
          {status === 'loading' && (
            <div className="hg-status">
              <span className="spin" /> Loading model
              <span className="hg-progress"><div style={{ width: `${progress}%` }} /></span>
              {progress}%
            </div>
          )}
          {status === 'error' && (
            <div className="hg-status" style={{ color: 'var(--miss)' }}>Failed to load model.</div>
          )}

          {/* ── Game ── */}
          {game && status === 'ready' && (
            <div className={`hg-stack ${mode === 'user' && !aiHintActive ? 'user-mode-only' : ''}`}>
              {/* LEFT COLUMN: Game */}
              <div className="hg-left">
                <div className="hg-board">
                  <div className="hg-stage">
                    <div className="hg-gallows"><Gallows wrong={game.wrong} /></div>

                    <div className="hg-meta">
                      {/* AI mode: reveal the target word so you can follow along */}
                      {mode === 'ai' && (
                        <div className="hg-answer">
                          Target word <b>{game.word}</b>
                        </div>
                      )}

                      {/* strike tracker (no hearts) */}
                      <div className="hg-strikes">
                        <span className="label">Wrong</span>
                        <span className="pips">
                          {Array.from({ length: 6 }, (_, i) => (
                            <span key={i} className="hg-pip" data-used={i < game.wrong} />
                          ))}
                        </span>
                        <span className="count">{game.wrong}/6</span>
                      </div>

                      {/* word tiles */}
                      <div className="hg-word" data-length={game.word.length >= 21 ? 'long' : game.word.length}>
                        {[...game.masked].map((c, i) => (
                          <div key={i} className="hg-tile" data-filled={c !== '_'} data-blank={c === '_'}>
                            {c !== '_' ? c : ''}
                          </div>
                        ))}
                      </div>

                      {/* result */}
                      {game.status === 'won' && (
                        <div className="hg-result" data-r="won">
                          {mode === 'ai' ? '✓ Model solved it' : '✓ You got it!'} — {game.history.length} guesses, {game.wrong} wrong
                        </div>
                      )}
                      {game.status === 'lost' && (
                        <div className="hg-result" data-r="lost">
                          ✗ {mode === 'ai' ? 'Model lost' : 'Out of guesses'} — the word was
                          <span className="word">{game.word}</span>
                        </div>
                      )}

                      {/* ── History ── */}
                      {game.history.length > 0 && (
                        <div className="hg-hist">
                          <div className="hg-hist-label">Guess order</div>
                          <div className="hg-chips">
                            {game.history.map((h, i) => (
                              <span key={i} className="hg-chip" data-hit={h.hit}>
                                {h.letter}{h.hit ? '✓' : '✗'}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* actions */}
                  <div className="hg-actions">
                    {mode === 'ai' && game.status === 'playing' && (
                      <>
                        <button className="hg-btn primary" onClick={step} disabled={thinking || auto}>
                          Step
                        </button>
                        <button className={`hg-btn ${auto ? 'stop' : 'go'}`} onClick={() => setAuto(a => !a)} disabled={thinking}>
                          {auto ? 'Pause' : 'Auto-play'}
                        </button>
                      </>
                    )}
                    <button className="hg-btn" onClick={startRandom}>New word</button>
                    {mode === 'ai' && (
                      <button className="hg-btn" onClick={() => setShowCustom(s => !s)}>Custom word</button>
                    )}
                    {mode === 'user' && game.status === 'playing' && (
                      <button className="hg-btn" onClick={() => setShowHint(h => !h)}>
                        {showHint ? 'Hide AI hint' : 'Show AI hint'}
                      </button>
                    )}
                  </div>

                  {showCustom && (
                    <div className="hg-custom">
                      <input
                        className="hg-input" value={custom} maxLength={50}
                        placeholder="type a word for it to guess…"
                        onChange={e => setCustom(e.target.value.replace(/[^a-zA-Z]/g, ''))}
                        onKeyDown={e => e.key === 'Enter' && startCustom()}
                      />
                      <button className="hg-btn primary" onClick={startCustom}>Set</button>
                    </div>
                  )}
                </div>

                {/* ── User keyboard ── */}
                {mode === 'user' && (
                  <div className="hg-kbd">
                    <div className="hg-kbd-rows">
                      {KBD_ROWS.map((row, ri) => (
                        <div className="hg-kbd-row" key={ri}>
                          {row.split('').map(l => {
                            const gd = game.guessed.has(l)
                            const st = gd ? (game.word.includes(l) ? 'hit' : 'miss') : 'idle'
                            return (
                              <button
                                key={l} className="hg-key" data-state={st}
                                data-hint={aiHintActive && l === next}
                                disabled={gd || game.status !== 'playing'}
                                onClick={() => guess(l)}
                              >{l}</button>
                            )
                          })}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT COLUMN: Model mind */}
              {/* Always show in AI mode, only show in User mode when hint is active */}
              {mode === 'ai' ? (
                <div className="hg-right">
                  <div className="hg-mind">
                    <div className="hg-mind-head">
                      <div className="hg-mind-title">
                        Model confidence
                        {next && game.status === 'playing' && <span className="chip">NEXT&nbsp;·&nbsp;{next.toUpperCase()}</span>}
                      </div>
                      {thinking && <span className="hg-think"><span className="spin" />inferring…</span>}
                    </div>
                    <div className="hg-bars">
                      {sortedLetters.map(l => {
                        const i = l.charCodeAt(0) - 97
                        const gd = game.guessed.has(l)
                        const done = gd ? (game.word.includes(l) ? 'hit' : 'miss') : undefined
                        const isNext = l === next && game.status === 'playing'
                        const pct = Math.round(probs[i] * 100)
                        return (
                          <div key={l} className="hg-bar" data-next={isNext} data-done={done}>
                            <span className="ltr">{l}</span>
                            <div className="hg-track">
                              <div className="hg-fill" style={{ width: `${gd ? 100 : Math.max(pct, 1.5)}%` }} />
                              {isNext && <span className="hg-tag">NEXT</span>}
                              <span className="pct">{gd ? (done === 'hit' ? '✓' : '✗') : `${pct}%`}</span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              ) : aiHintActive && (
                <div className="hg-right">
                  <div className="hg-mind">
                    <div className="hg-mind-head">
                      <div className="hg-mind-title">
                        AI Hint
                        {next && game.status === 'playing' && <span className="chip">NEXT&nbsp;·&nbsp;{next.toUpperCase()}</span>}
                      </div>
                      {thinking && <span className="hg-think"><span className="spin" />inferring…</span>}
                    </div>
                    <div className="hg-bars">
                      {sortedLetters.map(l => {
                        const i = l.charCodeAt(0) - 97
                        const gd = game.guessed.has(l)
                        const done = gd ? (game.word.includes(l) ? 'hit' : 'miss') : undefined
                        const isNext = l === next && game.status === 'playing'
                        const pct = Math.round(probs[i] * 100)
                        return (
                          <div key={l} className="hg-bar" data-next={isNext} data-done={done}>
                            <span className="ltr">{l}</span>
                            <div className="hg-track">
                              <div className="hg-fill" style={{ width: `${gd ? 100 : Math.max(pct, 1.5)}%` }} />
                              {isNext && <span className="hg-tag">NEXT</span>}
                              <span className="pct">{gd ? (done === 'hit' ? '✓' : '✗') : `${pct}%`}</span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
