import { useMemo, useState } from 'react'
import { CHART } from '../charts/chartKit'
import { visual } from '../../data/cliniverseResults'

/**
 * Candidate Damage Landscape.
 *
 * For each of the top candidate patterns, the spread of *measured* excess NLL
 * across 20 development resplits x 5 folds. Each dot is one held-out fold.
 *
 * This is a distribution of a loss difference per candidate pattern. It is NOT
 * a feature-attribution method and must never be described as one — no SHAP,
 * no importance ranking, no per-feature contribution exists in this project.
 */

const landscape = visual.damageLandscape
const FROZEN = 'BUN+Glucose+Na'

export default function DamageLandscape({
  height = 300,
  count = 8,
  compact = false,
}: {
  height?: number
  count?: number
  compact?: boolean
}) {
  const [hovered, setHovered] = useState<string | null>(null)
  const series = landscape.series.slice(0, count)

  const { rows, scale } = useMemo(() => {
    const all = series.flatMap((s) => s.values)
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    const pad = (hi - lo) * 0.06
    const min = lo - pad
    const max = hi + pad
    const x = (v: number) => ((v - min) / (max - min)) * 100

    return {
      rows: series.map((s) => {
        // deterministic vertical jitter: index-derived, never random
        const dots = s.values.map((v, i) => ({
          v,
          x: x(v),
          y: 50 + Math.sin(i * 2.399963) * 26,
        }))
        return {
          ...s,
          dots,
          xMean: x(s.mean),
          xP10: x(s.p10),
          xP90: x(s.p90),
          isFrozen: s.name === FROZEN,
        }
      }),
      scale: { min, max, zero: x(0) },
    }
  }, [series])

  const rowH = compact ? 22 : Math.max(26, Math.floor(height / series.length))

  return (
    <div className="vz-landscape">
      {rows.map((row) => {
        const on = hovered === row.name
        return (
          <div
            key={row.name}
            className={`vz-lane${row.isFrozen ? ' is-frozen' : ''}${on ? ' is-hovered' : ''}`}
            style={{ height: rowH }}
            onMouseEnter={() => setHovered(row.name)}
            onMouseLeave={() => setHovered(null)}
          >
            {!compact && (
              <span className="vz-lane-name" title={row.analytes.join(' + ')}>
                {row.analytes.join(' + ')}
              </span>
            )}
            <span className="vz-lane-plot">
              {/* zero reference */}
              <i className="vz-lane-zero" style={{ left: `${scale.zero}%` }} aria-hidden />
              {/* p10–p90 span */}
              <i
                className="vz-lane-span"
                style={{ left: `${row.xP10}%`, width: `${row.xP90 - row.xP10}%` }}
                aria-hidden
              />
              {row.dots.map((d, i) => (
                <i
                  key={i}
                  className="vz-dot"
                  style={{
                    left: `${d.x}%`,
                    top: `${d.y}%`,
                    background: d.v > 0 ? CHART.withheld : CHART.control,
                  }}
                  aria-hidden
                />
              ))}
              {/* mean marker */}
              <i className="vz-lane-mean" style={{ left: `${row.xMean}%` }} aria-hidden />
            </span>
            {!compact && (
              <span className="vz-lane-stat">
                <b>{row.mean.toFixed(5)}</b>
                <em>{(row.positiveFraction * 100).toFixed(0)}% &gt; 0</em>
              </span>
            )}
          </div>
        )
      })}

      {!compact && (
        <div className="vz-landscape-foot">
          <span className="cv-meta">
            each dot is one held-out fold · {landscape.nObservationsPerCandidate} per candidate ·{' '}
            {landscape.nCandidates} candidates enumerated
          </span>
          <span className="vz-landscape-key">
            <i style={{ background: CHART.withheld }} /> damage above control
            <i style={{ background: CHART.control, marginLeft: 10 }} /> below
          </span>
        </div>
      )}

      <p className="sr-only">
        Distribution of measured development excess NLL for the top {series.length} candidate
        patterns, {landscape.nObservationsPerCandidate} held-out folds each. A distribution of loss
        differences, not a feature-attribution or importance measure.
      </p>
    </div>
  )
}
