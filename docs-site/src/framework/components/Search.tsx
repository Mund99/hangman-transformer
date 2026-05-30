import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { NAV, type NavItem } from '../../nav'

interface Page { label: string; path: string; section: string }

function flatPages(items: NavItem[], section = ''): Page[] {
  return items.flatMap(item =>
    item.children
      ? flatPages(item.children, item.label)
      : item.path
      ? [{ label: item.label, path: item.path, section }]
      : []
  )
}

const PAGES = flatPages(NAV)

function Highlight({ text, query }: { text: string; query: string }) {
  const q = query.trim()
  if (!q) return <>{text}</>
  const i = text.toLowerCase().indexOf(q.toLowerCase())
  if (i === -1) return <>{text}</>
  return (
    <>
      {text.slice(0, i)}
      <mark className="search-mark">{text.slice(i, i + q.length)}</mark>
      {text.slice(i + q.length)}
    </>
  )
}

export default function Search() {
  const [query,  setQuery]  = useState('')
  const [active, setActive] = useState(0)
  const [open,   setOpen]   = useState(false)
  const wrapRef  = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const results = query.trim()
    ? PAGES.filter(p =>
        p.label.toLowerCase().includes(query.toLowerCase()) ||
        p.section.toLowerCase().includes(query.toLowerCase())
      )
    : []

  // ⌘K / Ctrl+K
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', fn)
    return () => window.removeEventListener('keydown', fn)
  }, [])

  // Click outside to close
  useEffect(() => {
    const fn = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', fn)
    return () => document.removeEventListener('mousedown', fn)
  }, [])

  const go = useCallback((path: string) => {
    navigate(path)
    setQuery('')
    setOpen(false)
    inputRef.current?.blur()
  }, [navigate])

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive(a => Math.min(a + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive(a => Math.max(a - 1, 0))
    } else if (e.key === 'Enter') {
      if (results[active]) go(results[active].path)
    } else if (e.key === 'Escape') {
      setOpen(false)
      setQuery('')
      inputRef.current?.blur()
    }
  }

  return (
    <div className="header-search" ref={wrapRef}>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      <input
        ref={inputRef}
        type="text"
        placeholder="Search docs…"
        value={query}
        autoComplete="off"
        onChange={e => { setQuery(e.target.value); setActive(0); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      <span className="kbd">⌘K</span>

      {open && results.length > 0 && (
        <div className="search-dropdown">
          {results.map((r, i) => (
            <div
              key={r.path}
              className={`search-result${i === active ? ' active' : ''}`}
              onMouseDown={() => go(r.path)}
              onMouseEnter={() => setActive(i)}
            >
              {r.section && <span className="search-result-section">{r.section}</span>}
              <span className="search-result-label">
                <Highlight text={r.label} query={query} />
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
