import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { AXIS, CHART } from '../charts/chartKit'
import { confirmation, visual, type PatientRow } from '../../data/cliniverseResults'

/**
 * Failure Slice.
 *
 * Every one of the 4,000 set-c patients, plotted as clean predicted risk
 * against that patient's excess NLL under the frozen pattern. Colour is the
 * recorded outcome. All six hover fields come straight from
 * `setc_oneshot_predictions.npz`; nothing is modelled or imputed here.
 */

const patients = visual.patients

export default function FailureSlice({
  height = 300,
  compact = false,
}: {
  height?: number
  compact?: boolean
}) {
  const [showSurvived, setShowSurvived] = useState(true)
  const [showDied, setShowDied] = useState(true)

  const { survived, died } = useMemo(
    () => ({
      survived: patients.filter((p) => p.y === 0),
      died: patients.filter((p) => p.y === 1),
    }),
    [],
  )

  return (
    <div className="vz-slice">
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 12, bottom: compact ? 4 : 16, left: -6 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="c"
            name="Clean predicted risk"
            domain={[0, 0.8]}
            ticks={[0, 0.2, 0.4, 0.6, 0.8]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
            label={
              compact
                ? undefined
                : { value: 'clean predicted risk', position: 'insideBottom', offset: -8, fontSize: 9.5, fill: CHART.axis }
            }
          />
          <YAxis
            type="number"
            dataKey="d"
            name="Excess NLL"
            domain={[-1, 1.6]}
            ticks={[-1, -0.5, 0, 0.5, 1, 1.5]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
            width={46}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <ZAxis range={[6, 6]} />
          <ReferenceLine y={0} stroke="#B7CCD4" strokeWidth={1.2} />
          <ReferenceLine
            y={confirmation.deltaC}
            stroke={CHART.navy}
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{
              value: `mean Δ C +${confirmation.deltaC.toFixed(4)}`,
              position: 'insideTopRight',
              fontSize: 9,
              fill: CHART.navy,
            }}
          />
          <Tooltip
            cursor={{ strokeDasharray: '3 3', stroke: '#B7CCD4' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const p = payload[0].payload as PatientRow
              const shift = p.w - p.c
              return (
                <div className="cv-chart-tip">
                  <strong>Record {p.id}</strong>
                  <div className="cv-tip-row">
                    <i style={{ background: p.y === 1 ? CHART.withheld : CHART.clean }} />
                    {p.y === 1 ? 'In-hospital death' : 'Survived'}
                  </div>
                  <div className="cv-tip-row">clean risk: {p.c.toFixed(5)}</div>
                  <div className="cv-tip-row">withheld risk: {p.w.toFixed(5)}</div>
                  <div className="cv-tip-row">
                    risk change: {shift > 0 ? '+' : '−'}
                    {Math.abs(shift).toFixed(5)}
                  </div>
                  <div className="cv-tip-row">removed cells: {p.rc}</div>
                  <div className="cv-tip-row">
                    excess NLL: {p.d > 0 ? '+' : '−'}
                    {Math.abs(p.d).toFixed(5)}
                  </div>
                </div>
              )
            }}
          />
          {showSurvived && (
            <Scatter
              name="Survived"
              data={survived}
              fill={CHART.clean}
              fillOpacity={0.3}
              isAnimationActive={false}
            />
          )}
          {showDied && (
            <Scatter
              name="Died"
              data={died}
              fill={CHART.withheld}
              fillOpacity={0.62}
              isAnimationActive={false}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>

      {!compact && (
        <div className="vz-slice-legend">
          <button
            type="button"
            className={`vz-toggle${showSurvived ? ' is-on' : ''}`}
            onClick={() => setShowSurvived((v) => !v)}
            aria-pressed={showSurvived}
          >
            <i style={{ background: CHART.clean }} /> Survived ({survived.length.toLocaleString('en-US')})
          </button>
          <button
            type="button"
            className={`vz-toggle${showDied ? ' is-on' : ''}`}
            onClick={() => setShowDied((v) => !v)}
            aria-pressed={showDied}
          >
            <i style={{ background: CHART.withheld }} /> In-hospital death ({died.length})
          </button>
          <span className="cv-meta">
            points above zero were damaged more than the amount-matched control
          </span>
        </div>
      )}

      <p className="sr-only">
        Scatter of all {patients.length.toLocaleString('en-US')} set-c patients: clean predicted
        risk against per-patient excess negative log-likelihood, coloured by recorded outcome.
        {visual.burden.nZero} patients had no eligible cells removed.
      </p>
    </div>
  )
}
