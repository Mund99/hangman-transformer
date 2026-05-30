import { type ReactNode } from 'react'

type CalloutType = 'note' | 'tip' | 'warning' | 'danger' | 'info' | 'quote'

const ICONS: Record<CalloutType, string> = {
  note: 'ℹ️', tip: '💡', warning: '⚠️', danger: '🔴', info: '📌', quote: '💬',
}

interface Props {
  type?: CalloutType
  title?: string
  children: ReactNode
}

export default function Callout({ type = 'note', title, children }: Props) {
  return (
    <div className={`callout ${type}`}>
      <div className="callout-title">
        <span>{ICONS[type]}</span>
        <span>{title ?? type.charAt(0).toUpperCase() + type.slice(1)}</span>
      </div>
      {children}
    </div>
  )
}
