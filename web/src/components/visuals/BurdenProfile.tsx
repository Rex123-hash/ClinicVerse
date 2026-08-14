import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AXIS, CHART, ChartTooltip } from '../charts/chartKit'
import { visual } from '../../data/cliniverseResults'

/**
 * Withholding Burden Profile.
 *
 * How many cells the frozen pattern actually removed, per patient. Removal is
 * clipped to what was naturally observed, so the burden is uneven — and 94
 * patients had nothing eligible to remove at all.
 */

const burden = visual.burden

export default function BurdenProfile({
  height = 300,
  compact = false,
}: {
  height?: number
  compact?: boolean
}) {
  const data = burden.histogram

  return (
    <div className="vz-burden">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 10, right: 12, bottom: compact ? 0 : 14, left: -8 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="removed"
            tick={{ ...AXIS.tick, fontSize: 9 }}
            tickLine={false}
            axisLine={AXIS.line}
            interval={compact ? 4 : 1}
            label={
              compact
                ? undefined
                : {
                    value: 'cells removed per patient',
                    position: 'insideBottom',
                    offset: -6,
                    fontSize: 9.5,
                    fill: CHART.axis,
                  }
            }
          />
          <YAxis tick={AXIS.tick} tickLine={false} axisLine={false} width={48} />
          <Tooltip
            cursor={{ fill: 'rgba(16,170,165,0.05)' }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={`${payload[0].payload.removed} cells removed`}
                  rows={[
                    {
                      name: 'Patients',
                      value: Number(payload[0].value).toLocaleString('en-US'),
                      color: CHART.clean,
                    },
                  ]}
                  note={
                    payload[0].payload.removed === 0
                      ? 'no eligible cells for this patient'
                      : undefined
                  }
                />
              ) : null
            }
          />
          <ReferenceLine
            x={burden.median}
            stroke={CHART.navy}
            strokeDasharray="4 4"
            strokeWidth={1.2}
            label={{ value: `median ${burden.median}`, position: 'top', fontSize: 9, fill: CHART.navy }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]} animationDuration={800}>
            {data.map((row) => (
              <Cell
                key={row.removed}
                fill={
                  row.removed === 0
                    ? '#C6D5DB'
                    : row.removed >= burden.p10 && row.removed <= burden.p90
                      ? CHART.clean
                      : '#9AD6D2'
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {!compact && (
        <div className="vz-burden-stats">
          <Stat k="mean" v={burden.mean.toFixed(5)} />
          <Stat k="median" v={String(burden.median)} />
          <Stat k="p10" v={String(burden.p10)} />
          <Stat k="p90" v={String(burden.p90)} />
          <Stat k="max" v={String(burden.max)} />
          <Stat k="zero removed" v={String(burden.nZero)} tone="muted" />
        </div>
      )}

      <p className="sr-only">
        Distribution of removed cells per patient: mean {burden.mean.toFixed(5)}, median{' '}
        {burden.median}, p10 {burden.p10}, p90 {burden.p90}, maximum {burden.max}.{' '}
        {burden.nZero} patients had no eligible cells to remove.
      </p>
    </div>
  )
}

function Stat({ k, v, tone }: { k: string; v: string; tone?: 'muted' }) {
  return (
    <div className={`vz-stat${tone ? ' is-muted' : ''}`}>
      <span>{k}</span>
      <strong>{v}</strong>
    </div>
  )
}
