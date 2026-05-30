type BadgeType = 'python' | 'ts' | 'js' | 'electron' | 'active' | 'archived' | string

interface Props { type: BadgeType; label?: string }

export default function Badge({ type, label }: Props) {
  return <span className={`badge ${type}`}>{label ?? type}</span>
}
