import { useState, useEffect, useContext } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Sidebar, Search, ThemeContext } from './framework'
import { siteConfig } from './site.config'
import './framework/styles/globals.css'

// ── Pages — add your imports here ────────────────────────
import HomePage from './pages/HomePage'
import Play from './pages/Play'
import Data from './pages/docs/Data'
import Baselines from './pages/docs/Baselines'
import Architecture from './pages/docs/Architecture'
import Training from './pages/docs/Training'
import Analysis from './pages/docs/Analysis'
import Experiments from './pages/docs/Experiments'
import Pipeline from './pages/docs/Pipeline'
import Glossary from './pages/docs/Glossary'
// ─────────────────────────────────────────────────────────


function ThemeToggle() {
  const { theme, toggle } = useContext(ThemeContext)
  return (
    <button className="theme-toggle" onClick={toggle} aria-label="Toggle theme">
      {theme === 'dark' ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      )}
    </button>
  )
}

function Header({ onMenu }: { onMenu: () => void }) {
  return (
    <header className="header">
      <button onClick={onMenu} className="mobile-btn" aria-label="Menu">☰</button>

      <span className="header-title">{siteConfig.title}</span>

      <div className="header-spacer" />

      <Search />

      <a
        href={siteConfig.github}
        target="_blank"
        rel="noopener noreferrer"
        className="header-github"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        <span>GitHub</span>
      </a>

      <ThemeToggle />
    </header>
  )
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    (localStorage.getItem('theme') as 'light' | 'dark') || 'light'
  )

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  // Sync browser tab title from site.config.ts
  useEffect(() => {
    document.title = siteConfig.title
  }, [])

  // Apply brand accent colour from site.config.ts
  useEffect(() => {
    if (siteConfig.accentColor) {
      document.documentElement.style.setProperty('--accent', siteConfig.accentColor)
      document.documentElement.style.setProperty('--accent-light', siteConfig.accentColor)
      // Logo bg is set once here — not inside [data-theme="dark"] so it never changes with theme
      document.documentElement.style.setProperty('--logo-bg', siteConfig.accentColor)
    }
  }, [])

  useEffect(() => {
    const onResize = () => { if (window.innerWidth < 768) setSidebarOpen(false) }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  return (
    <ThemeContext.Provider value={{ theme, toggle: () => setTheme(t => t === 'light' ? 'dark' : 'light') }}>
      <div className="shell">
        <Header onMenu={() => setSidebarOpen(o => !o)} />
        <div className="shell-body">
          <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

          {sidebarOpen && (
            <div
              onClick={() => setSidebarOpen(false)}
              style={{ position: 'fixed', inset: 0, zIndex: 90, background: 'rgba(0,0,0,0.25)', top: 'var(--header-h)' }}
            />
          )}

          {/* ── Add your routes here ───────────────────────── */}
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/play" element={<Play />} />
            <Route path="/docs/data" element={<Data />} />
            <Route path="/docs/baselines" element={<Baselines />} />
            <Route path="/docs/architecture" element={<Architecture />} />
            <Route path="/docs/training" element={<Training />} />
            <Route path="/docs/analysis" element={<Analysis />} />
            <Route path="/docs/experiments" element={<Experiments />} />
            <Route path="/docs/pipeline" element={<Pipeline />} />
            <Route path="/docs/glossary" element={<Glossary />} />
          </Routes>
        </div>
      </div>
    </ThemeContext.Provider>
  )
}
