import type { ReactNode } from 'react'

/** One colour convention for the whole application. */
export const CHART = {
  clean: '#0E9B96',
  cleanSoft: 'rgba(14, 155, 150, 0.16)',
  withheld: '#FF6A16',
  withheldSoft: 'rgba(255, 106, 22, 0.15)',
  control: '#8D99AE',
  grid: '#EEF3F5',
  axis: '#8D99AE',
  navy: '#132A52',
  success: '#1AA66E',
} as const

export const AXIS = {
  tick: { fontSize: 10.5, fill: CHART.axis },
  line: { stroke: CHART.grid },
} as const

interface TipRow {
  name: string
  value: string
  color: string
}

export function ChartTooltip({
  title,
  rows,
  note,
}: {
  title: ReactNode
  rows: readonly TipRow[]
  note?: string
}) {
  return (
    <div className="cv-chart-tip">
      <strong>{title}</strong>
      {rows.map((row) => (
        <div className="cv-tip-row" key={row.name}>
          <i style={{ background: row.color }} />
          {row.name}: {row.value}
        </div>
      ))}
      {note && <div style={{ marginTop: 4, opacity: 0.7 }}>{note}</div>}
    </div>
  )
}

export function Legend({
  items,
}: {
  items: readonly { label: string; color: string; square?: boolean }[]
}) {
  return (
    <div className="cv-legend">
      {items.map((item) => (
        <span key={item.label}>
          <i className={item.square ? 'sq' : ''} style={{ background: item.color }} />
          {item.label}
        </span>
      ))}
    </div>
  )
}

/**
 * A text alternative for a chart. Charts are images to a screen reader; this
 * gives the same conclusion in words.
 */
export function ChartDescription({ children }: { children: ReactNode }) {
  return <p className="sr-only">{children}</p>
}
