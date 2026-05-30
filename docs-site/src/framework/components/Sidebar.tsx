import { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { NAV, type NavItem } from '../../nav'

function Section({ item }: { item: NavItem }) {
  const [open, setOpen] = useState(true)

  return (
    <div className="nav-section">
      <div
        className="nav-section-hd"
        onClick={() => setOpen(o => !o)}
        role="button"
        tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && setOpen(o => !o)}
      >
        <span>{item.label}</span>
        <span className={`nav-chevron ${open ? 'open' : ''}`}>▶</span>
      </div>
      {open && item.children?.map(child => (
        <NavLink
          key={child.path}
          to={child.path!}
          end
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          {child.label}
        </NavLink>
      ))}
    </div>
  )
}

export default function Sidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem('sidebar-collapsed') === 'true'
  )

  useEffect(() => {
    localStorage.setItem('sidebar-collapsed', String(collapsed))
    if (collapsed) {
      document.documentElement.setAttribute('data-sidebar-collapsed', '')
    } else {
      document.documentElement.removeAttribute('data-sidebar-collapsed')
    }
  }, [collapsed])

  // Sync attribute on first render
  useEffect(() => {
    if (collapsed) document.documentElement.setAttribute('data-sidebar-collapsed', '')
    return () => document.documentElement.removeAttribute('data-sidebar-collapsed')
  }, [])

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''} ${collapsed ? 'collapsed' : ''}`} onClick={e => {
      if ((e.target as HTMLElement).tagName === 'A') onClose()
    }}>
      <div className="sidebar-inner">
        {NAV.map((item, i) =>
          item.children
            ? <Section key={i} item={item} />
            : (
              <NavLink
                key={i}
                to={item.path!}
                end
                className={({ isActive }) => `nav-top-link${isActive ? ' active' : ''}`}
              >
                {item.label}
              </NavLink>
            )
        )}
      </div>
      <button
        className="sidebar-toggle"
        onClick={e => { e.stopPropagation(); setCollapsed(c => !c) }}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? '›' : '‹'}
      </button>
    </aside>
  )
}
