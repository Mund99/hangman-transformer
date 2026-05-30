import { useState, useEffect, useContext } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Sidebar, Search, ThemeContext } from './framework'
import { siteConfig } from './site.config'
import './framework/styles/globals.css'

// ── Pages — add your imports here ────────────────────────
import Play from './pages/Play'
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
            <Route path="/" element={<Play />} />
          </Routes>
        </div>
      </div>
    </ThemeContext.Provider>
  )
}
