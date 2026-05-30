import { useEffect, useRef, useState } from 'react'

interface Props { children: string }

let _id = 0
let initialized = false

const THEME_VARS = {
  primaryColor:        '#dbeafe',
  primaryTextColor:    '#1e3a8a',
  primaryBorderColor:  '#93c5fd',
  lineColor:           '#3b5bdb',
  secondaryColor:      '#eff6ff',
  tertiaryColor:       '#f0f9ff',
  edgeLabelBackground: '#eef2ff',
  fontFamily:          "'Outfit', sans-serif",
  fontSize:            '13px',
  clusterBkg:          '#f8faff',
  clusterBorder:       '#c7d2fe',
  titleColor:          '#1e3a8a',
}

async function getMermaid() {
  const { default: m } = await import('mermaid')
  if (!initialized) {
    m.initialize({
      startOnLoad: false,
      theme: 'base',
      themeVariables: THEME_VARS,
      flowchart: { curve: 'basis', htmlLabels: false },
    })
    initialized = true
  }
  return m
}

export default function Diagram({ children }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const id = useRef(`mmd-${++_id}`)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    ;(async () => {
      try {
        const m = await getMermaid()
        if (cancelled) return

        // Remove any stale Mermaid helper element left from a previous render
        document.getElementById(id.current)?.remove()
        if (ref.current) ref.current.innerHTML = ''

        const { svg } = await m.render(id.current, children.trim())

        if (!cancelled && ref.current) ref.current.innerHTML = svg
      } catch (e) {
        if (!cancelled) setErr(String(e))
      }
    })()

    return () => { cancelled = true }
  }, [children])

  if (err) return (
    <div className="diagram-wrap" style={{ textAlign: 'left' }}>
      <p style={{ color: '#ef4444', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
        Diagram error: {err}
      </p>
    </div>
  )

  return <div className="diagram-wrap" ref={ref} />
}
