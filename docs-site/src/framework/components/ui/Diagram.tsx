import { useEffect, useRef, useState } from 'react'

interface Props { children: string }

let _id = 0

const LIGHT_VARS = {
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

const DARK_VARS = {
  primaryColor:        '#1e3a5f',
  primaryTextColor:    '#93c5fd',
  primaryBorderColor:  '#3b82f6',
  lineColor:           '#60a5fa',
  secondaryColor:      '#1a2030',
  tertiaryColor:       '#162030',
  edgeLabelBackground: '#1a2030',
  fontFamily:          "'Outfit', sans-serif",
  fontSize:            '13px',
  clusterBkg:          '#161e2e',
  clusterBorder:       '#2d4a7a',
  titleColor:          '#93c5fd',
}

function currentTheme() {
  return document.documentElement.dataset.theme ?? 'light'
}

async function renderDiagram(id: string, src: string) {
  const { default: m } = await import('mermaid')
  m.initialize({
    startOnLoad: false,
    theme: 'base',
    themeVariables: currentTheme() === 'dark' ? DARK_VARS : LIGHT_VARS,
    flowchart: { curve: 'basis', htmlLabels: true },
  })
  return m.render(id, src.trim())
}

export default function Diagram({ children }: Props) {
  const ref  = useRef<HTMLDivElement>(null)
  const id   = useRef(`mmd-${++_id}`)
  const [err, setErr]     = useState<string | null>(null)
  const [theme, setTheme] = useState<string>(currentTheme)

  // Re-render whenever the data-theme attribute changes
  useEffect(() => {
    const obs = new MutationObserver(() => setTheme(currentTheme()))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => obs.disconnect()
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        document.getElementById(id.current)?.remove()
        if (ref.current) ref.current.innerHTML = ''

        const { svg } = await renderDiagram(id.current, children)
        if (!cancelled && ref.current) ref.current.innerHTML = svg
      } catch (e) {
        if (!cancelled) setErr(String(e))
      }
    })()
    return () => { cancelled = true }
  }, [children, theme])

  if (err) return (
    <div className="diagram-wrap" style={{ textAlign: 'left' }}>
      <p style={{ color: '#ef4444', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
        Diagram error: {err}
      </p>
    </div>
  )

  return <div className="diagram-wrap" ref={ref} />
}
