import { useState, Children, isValidElement, type ReactNode } from 'react'

// ── Tab (individual panel) ─────────────────────────────────────────────────

export interface TabProps {
  /** Label shown in the tab bar */
  label: string
  children: ReactNode
}

/** Individual tab panel — wrap content inside <Tabs> */
export function Tab({ children }: TabProps) {
  return <>{children}</>
}

// ── Tabs (container) ───────────────────────────────────────────────────────

interface TabsProps {
  children: ReactNode
}

/**
 * Tabbed content container.
 *
 * ```tsx
 * <Tabs>
 *   <Tab label="npm">
 *     <CodeBlock language="bash">npm install my-pkg</CodeBlock>
 *   </Tab>
 *   <Tab label="yarn">
 *     <CodeBlock language="bash">yarn add my-pkg</CodeBlock>
 *   </Tab>
 * </Tabs>
 * ```
 */
export default function Tabs({ children }: TabsProps) {
  const [active, setActive] = useState(0)

  const tabs = Children.toArray(children).filter(
    (child): child is React.ReactElement<TabProps> =>
      isValidElement(child) && typeof (child.props as TabProps).label === 'string'
  )

  if (tabs.length === 0) return null

  return (
    <div className="tabs">
      <div className="tabs-bar" role="tablist">
        {tabs.map((tab, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={i === active}
            className={`tab-btn${i === active ? ' active' : ''}`}
            onClick={() => setActive(i)}
          >
            {(tab.props as TabProps).label}
          </button>
        ))}
      </div>
      <div className="tab-content" role="tabpanel">
        {tabs[active]}
      </div>
    </div>
  )
}
