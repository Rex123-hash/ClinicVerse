import { useMemo, useState } from 'react'
import { charts, visual } from '../../data/cliniverseResults'

/**
 * Failure Concentration Surface.
 *
 * Mean per-patient excess NLL over a grid of clean predicted risk x removed
 * cell count, drawn as an isometric surface in plain SVG.
 *
 * This is a DESCRIPTIVE stress-test surface. Both axes are artefacts of the
 * experiment — a model output and a count of cells the mechanism removed — so
 * it shows where the measured damage concentrated. It is not a feature-response
 * relationship, and nothing here implies a biological mechanism.
 */

const surface = visual.concentrationSurface

/**
 * Diverging ramp centred on ZERO excess: teal below the amount-matched
 * control, near-white at parity, orange above it. Centring on the midpoint of
 * the data range instead would put "no damage" at an arbitrary colour.
 */
function ramp(t: number): string {
  const stops: [number, [number, number, number]][] = [
    [0, [14, 155, 150]],
    [0.42, [154, 214, 210]],
    [0.55, [238, 240, 236]],
    [0.72, [255, 176, 106]],
    [1, [214, 74, 6]],
  ]
  const clamped = Math.max(0, Math.min(1, t))
  for (let i = 1; i < stops.length; i += 1) {
    const [p1, c1] = stops[i - 1]
    const [p2, c2] = stops[i]
    if (clamped <= p2) {
      const k = (clamped - p1) / (p2 - p1 || 1)
      const rgb = c1.map((c, j) => Math.round(c + (c2[j] - c) * k))
      return `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})`
    }
  }
  return `rgb(${stops[stops.length - 1][1].join(' ')})`
}

interface Hover {
  risk: number
  removed: number
  meanExcess: number
  n: number
}

export default function ConcentrationSurface({
  width = 520,
  height = 320,
  compact = false,
}: {
  width?: number
  height?: number
  compact?: boolean
}) {
  const [hover, setHover] = useState<Hover | null>(null)

  const { quads, axes } = useMemo(() => {
    const { riskBins, cellBins, cells, zMin, zMax } = surface
    const at = (a: number, b: number) =>
      cells.find((c) => c.riskBin === a && c.cellBin === b) ?? null

    // corner heights, averaged from the (up to four) adjacent populated cells
    const corner = (a: number, b: number): number | null => {
      const vals: number[] = []
      for (const [da, db] of [
        [0, 0],
        [-1, 0],
        [0, -1],
        [-1, -1],
      ]) {
        const c = at(a + da, b + db)
        if (c && c.meanExcess !== null) vals.push(c.meanExcess)
      }
      return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
    }

    // colour is normalised about zero so parity with the control reads neutral
    const absMax = Math.max(Math.abs(zMin), Math.abs(zMax)) || 1
    const shade = (z: number) => 0.5 + (z / absMax) * 0.5
    const maxN = Math.max(...cells.map((cell) => cell.n), 1)
    // height stays a plain min-max ramp so the relief uses the full box
    const span = zMax - zMin || 1
    const norm = (z: number) => (z - zMin) / span

    // Fit the isometric projection to the available box: derive the tile size
    // from the grid extent rather than hard-coding it, or the surface spills
    // out of a small thumbnail.
    const rows = riskBins + cellBins
    const liftFrac = compact ? 0.26 : 0.3
    const lift = height * liftFrac
    const tileByHeight = (height * 0.9 - lift) / (rows * 0.5)
    const tileByWidth = (width * 0.92) / (rows * 0.87)
    const tile = Math.max(3, Math.min(tileByHeight, tileByWidth))

    const ox = width / 2
    const oy = height * 0.5 - (rows * tile * 0.5) / 2 + lift * 0.34

    const project = (a: number, b: number, z: number) => {
      const x = ox + (a - b) * tile * 0.87
      const y = oy + (a + b) * tile * 0.5 - norm(z) * lift
      return [x, y] as const
    }

    const list: {
      key: string
      d: string
      fill: string
      opacity: number
      depth: number
      data: Hover | null
    }[] = []

    for (let a = 0; a < riskBins; a += 1) {
      for (let b = 0; b < cellBins; b += 1) {
        const c = at(a, b)
        if (!c || c.meanExcess === null) continue
        const z00 = corner(a, b) ?? c.meanExcess
        const z10 = corner(a + 1, b) ?? c.meanExcess
        const z11 = corner(a + 1, b + 1) ?? c.meanExcess
        const z01 = corner(a, b + 1) ?? c.meanExcess
        const p00 = project(a, b, z00)
        const p10 = project(a + 1, b, z10)
        const p11 = project(a + 1, b + 1, z11)
        const p01 = project(a, b + 1, z01)
        list.push({
          key: `${a}-${b}`,
          d: `M${p00[0].toFixed(1)} ${p00[1].toFixed(1)} L${p10[0].toFixed(1)} ${p10[1].toFixed(1)} L${p11[0].toFixed(1)} ${p11[1].toFixed(1)} L${p01[0].toFixed(1)} ${p01[1].toFixed(1)} Z`,
          fill: ramp(shade(c.meanExcess)),
          // Sparse bins remain inspectable but cannot visually dominate a
          // descriptive surface built from much denser neighbouring bins.
          opacity: 0.62 + 0.38 * Math.sqrt(c.n / maxN),
          depth: a + b,
          data: {
            risk: c.risk,
            removed: c.removed,
            meanExcess: c.meanExcess,
            n: c.n,
          },
        })
      }
    }
    // painter's algorithm: far tiles first
    list.sort((p, q) => p.depth - q.depth)

    return {
      quads: list,
      axes: {
        origin: project(0, 0, zMin),
        riskEnd: project(riskBins, 0, zMin),
        cellEnd: project(0, cellBins, zMin),
      },
    }
  }, [width, height, compact])

  return (
    <div className="vz-surface">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        role="img"
        aria-label={`Failure concentration surface: mean excess NLL over clean predicted risk and removed cell count, ranging ${surface.zMin} to ${surface.zMax}; sparse bins are de-emphasized and each bin exposes its sample size`}
        onMouseLeave={() => setHover(null)}
      >
        {/* base axes */}
        <g stroke="#CBD9DF" strokeWidth="1" fill="none">
          <path d={`M${axes.origin[0]} ${axes.origin[1]} L${axes.riskEnd[0]} ${axes.riskEnd[1]}`} />
          <path d={`M${axes.origin[0]} ${axes.origin[1]} L${axes.cellEnd[0]} ${axes.cellEnd[1]}`} />
        </g>

        {quads.map((q) => (
          <path
            key={q.key}
            d={q.d}
            fill={q.fill}
            opacity={q.opacity}
            stroke="rgba(255,255,255,0.35)"
            strokeWidth="0.5"
            onMouseEnter={() => setHover(q.data)}
            style={{ cursor: 'crosshair' }}
          />
        ))}

        {!compact && (
          <>
            <text
              x={axes.riskEnd[0]}
              y={axes.riskEnd[1] + 16}
              fontSize="9.5"
              fill="#8D99AE"
              textAnchor="middle"
            >
              clean predicted risk →
            </text>
            <text
              x={axes.cellEnd[0]}
              y={axes.cellEnd[1] + 16}
              fontSize="9.5"
              fill="#8D99AE"
              textAnchor="middle"
            >
              ← removed cells
            </text>
          </>
        )}
      </svg>

      {!compact && (
        <div className="vz-surface-side">
          <div className="vz-legend-ramp" aria-hidden>
            <span
              style={{
                background: `linear-gradient(to top, ${ramp(0)}, ${ramp(0.5)}, ${ramp(1)})`,
              }}
            />
            <b>+{Math.max(Math.abs(surface.zMin), Math.abs(surface.zMax)).toFixed(2)}</b>
            <em>mean excess NLL · 0 = parity with control</em>
            <b>−{Math.max(Math.abs(surface.zMin), Math.abs(surface.zMax)).toFixed(2)}</b>
          </div>
          <div className={`vz-readout${hover ? ' is-on' : ''}`}>
            {hover ? (
              <>
                <span>clean risk ≈ {hover.risk.toFixed(3)}</span>
                <span>removed ≈ {hover.removed.toFixed(1)} cells</span>
                <b>excess NLL {hover.meanExcess > 0 ? '+' : ''}{hover.meanExcess.toFixed(4)}</b>
                <em>{hover.n} patients in bin</em>
              </>
            ) : (
              <em>Hover a tile for its bin</em>
            )}
          </div>
        </div>
      )}

      <p className="sr-only">
        Descriptive stress-test surface built from {charts.setc.removedCells.mean.toFixed(2)} mean
        removed cells per patient across 4,000 set-c patients. Sparse bins are visually de-emphasized;
        the inspector reports the patient count for every populated bin. Not a feature-response relationship.
      </p>
    </div>
  )
}
