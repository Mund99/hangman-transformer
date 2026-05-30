import { type ReactNode, useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { NAV, FLAT_PAGES } from '../../nav'
import { siteConfig } from '../../site.config'
import TableOfContents from './TableOfContents'

function ProgressBar() {
  const [pct, setPct] = useState(0)
  useEffect(() => {
    const el = document.querySelector('.main')
    if (!el) return
    const fn = () => {
      const { scrollTop, scrollHeight, clientHeight } = el as HTMLElement
      setPct(scrollHeight <= clientHeight ? 100 : (scrollTop / (scrollHeight - clientHeight)) * 100)
    }
    el.addEventListener('scroll', fn)
    return () => el.removeEventListener('scroll', fn)
  }, [])
  return <div className="progress" style={{ transform: `scaleX(${pct / 100})` }} />
}

function BackTop() {
  const [show, setShow] = useState(false)
  useEffect(() => {
    const el = document.querySelector('.main')
    if (!el) return
    const fn = () => setShow((el as HTMLElement).scrollTop > 320)
    el.addEventListener('scroll', fn)
    return () => el.removeEventListener('scroll', fn)
  }, [])
  return (
    <button
      className={`back-top ${show ? 'show' : ''}`}
      onClick={() => document.querySelector('.main')?.scrollTo({ top: 0, behavior: 'smooth' })}
    >↑</button>
  )
}

function useBreadcrumb(pathname: string) {
  for (const item of NAV) {
    if (item.children) {
      const child = item.children.find(c => c.path === pathname)
      if (child) return { section: item.label, label: child.label }
    }
  }
  return null
}

interface Props {
  children: ReactNode
  /** Relative path to this file from the src/ root — enables "Edit this page" link when siteConfig.editUrl is set.
   *  Example: editPath="pages/docs/MyPage.tsx"
   */
  editPath?: string
}

export default function DocLayout({ children, editPath }: Props) {
  const { pathname } = useLocation()

  useEffect(() => {
    document.querySelector('.main')?.scrollTo({ top: 0 })
  }, [pathname])

  const breadcrumb = useBreadcrumb(pathname)
  const idx = FLAT_PAGES.findIndex(p => p.path === pathname)
  const prev = idx > 0 ? FLAT_PAGES[idx - 1] : null
  const next = idx < FLAT_PAGES.length - 1 ? FLAT_PAGES[idx + 1] : null
  const editHref = siteConfig.editUrl && editPath
    ? `${siteConfig.editUrl.replace(/\/$/, '')}/${editPath.replace(/^\//, '')}`
    : null

  return (
    <>
      <ProgressBar />
      <div className="doc-wrapper">
        <div className="main">
          <div className="page">
            {breadcrumb && (
              <nav className="breadcrumb" aria-label="Breadcrumb">
                <span className="breadcrumb-section">{breadcrumb.section}</span>
                <span className="breadcrumb-sep">›</span>
                <span className="breadcrumb-current">{breadcrumb.label}</span>
              </nav>
            )}
            {children}
            {editHref && (
              <a href={editHref} className="edit-link" target="_blank" rel="noopener noreferrer">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                Edit this page
              </a>
            )}
            {(prev || next) && (
              <nav className="page-nav">
                <div className="page-nav-prev">
                  {prev && (
                    <Link to={prev.path} className="page-nav-btn">
                      <span className="page-nav-arrow">←</span>
                      <span className="page-nav-text">
                        <span className="page-nav-hint">Previous</span>
                        <span className="page-nav-label">{prev.label}</span>
                      </span>
                    </Link>
                  )}
                </div>
                <div className="page-nav-next">
                  {next && (
                    <Link to={next.path} className="page-nav-btn page-nav-btn--next">
                      <span className="page-nav-text">
                        <span className="page-nav-hint">Next</span>
                        <span className="page-nav-label">{next.label}</span>
                      </span>
                      <span className="page-nav-arrow">→</span>
                    </Link>
                  )}
                </div>
              </nav>
            )}
          </div>
        </div>
        <TableOfContents />
      </div>
      <BackTop />
    </>
  )
}
