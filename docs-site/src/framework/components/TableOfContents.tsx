import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'

interface Heading { id: string; text: string; level: number }

function slugify(text: string) {
  return text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-').trim()
}

export default function TableOfContents() {
  const { pathname } = useLocation()
  const [headings, setHeadings] = useState<Heading[]>([])
  const [activeId, setActiveId] = useState('')
  const observerRef = useRef<IntersectionObserver | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => {
      const els = Array.from(document.querySelectorAll('.page h2, .page h3')) as HTMLElement[]
      const seen = new Set<string>()
      const result: Heading[] = []
      els.forEach(el => {
        const text = el.textContent?.trim() || ''
        let id = slugify(text)
        if (!id) return
        let n = 0; const base = id
        while (seen.has(id)) { n++; id = `${base}-${n}` }
        seen.add(id)
        el.id = id
        result.push({ id, text, level: el.tagName === 'H2' ? 2 : 3 })
      })
      setHeadings(result)
      setActiveId(result[0]?.id ?? '')
    }, 80)
    return () => clearTimeout(timer)
  }, [pathname])

  useEffect(() => {
    if (!headings.length) return
    observerRef.current?.disconnect()
    const root = document.querySelector('.main') as HTMLElement | null
    observerRef.current = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) { setActiveId(entry.target.id); break }
        }
      },
      { root, rootMargin: '-8% 0px -72% 0px', threshold: 0 }
    )
    headings.forEach(h => {
      const el = document.getElementById(h.id)
      if (el) observerRef.current!.observe(el)
    })
    return () => observerRef.current?.disconnect()
  }, [headings])

  if (!headings.length) return <aside className="toc toc--empty" />

  return (
    <aside className="toc">
      <p className="toc-title">On this page</p>
      <ul className="toc-list">
        {headings.map(h => (
          <li key={h.id} className={`toc-item toc-h${h.level}`}>
            <a
              href={`#${h.id}`}
              className={`toc-link${activeId === h.id ? ' active' : ''}`}
              onClick={e => {
                e.preventDefault()
                document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                setActiveId(h.id)
              }}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </aside>
  )
}
