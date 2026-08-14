import type { ReactNode } from 'react'
import { useCountUp } from '../../hooks/useCountUp'

export function CountUp({
  value,
  places = 4,
  prefix = '',
  suffix = '',
}: {
  value: number
  places?: number
  prefix?: string
  suffix?: string
}) {
  const { ref, text } = useCountUp(value, places)
  return (
    <span ref={ref} className="cv-num">
      {prefix}
      {text}
      {suffix}
    </span>
  )
}

export function MetricTile({
  label,
  value,
  foot,
  tone = 'default',
  hint,
}: {
  label: ReactNode
  value: ReactNode
  foot?: ReactNode
  tone?: 'default' | 'teal' | 'orange' | 'green'
  hint?: string
}) {
  return (
    <div className={`cv-tile cv-tile--${tone}`} title={hint}>
      <span className="cv-tile-label">{label}</span>
      <span className="cv-tile-value">{value}</span>
      {foot && <span className="cv-tile-foot">{foot}</span>}
    </div>
  )
}
