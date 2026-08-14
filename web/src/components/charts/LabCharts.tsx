import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AXIS, CHART, ChartDescription, ChartTooltip } from './chartKit'
import { charts, confirmation, development } from '../../data/cliniverseResults'

const CONDITION_LABEL: Record<string, string> = {
  cell_random: 'Cell-random',
  group_structured: 'Group-structured',
  variable_matched_scattered: 'Variable-matched',
  none: 'Clean',
}

const CONDITION_COLOR: Record<string, string> = {
  cell_random: CHART.control,
  group_structured: CHART.withheld,
  variable_matched_scattered: CHART.clean,
}

/* ---------------------------------------------------------------------- */
/* M3 severity sweep — the executed performance-vs-stress evidence         */
/* ---------------------------------------------------------------------- */

export function SeverityResponseChart({
  metric = 'auroc',
  height = 150,
}: {
  metric?: 'auroc' | 'calibrationIntercept'
  height?: number
}) {
  const series = charts.stressResponse.series
  const clean = series.find((s) => s.condition === 'none')
  const severities = [0, 0.25, 0.5, 0.75]

  const data = severities.map((severity) => {
    const row: Record<string, number> = { severity }
    for (const condition of Object.keys(CONDITION_COLOR)) {
      const match =
        severity === 0
          ? clean
          : series.find((s) => s.condition === condition && s.severity === severity)
      if (match) row[condition] = match[metric]
    }
    return row
  })

  const isAuroc = metric === 'auroc'

  return (
    <>
      <ChartDescription>
        M3 severity sweep on development data. As withholding severity rises,{' '}
        {isAuroc
          ? 'AUROC falls modestly under every condition'
          : 'the calibration intercept drifts upward fastest under group-structured removal'}
        .
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="severity"
            type="number"
            domain={[0, 0.75]}
            ticks={severities}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
          />
          <YAxis
            domain={isAuroc ? [0.75, 0.85] : [-0.2, 0.7]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
            width={42}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={`Severity ${Number(label).toFixed(2)}`}
                  rows={payload.map((p) => ({
                    name: CONDITION_LABEL[String(p.dataKey)] ?? String(p.dataKey),
                    value: Number(p.value).toFixed(4),
                    color: String(p.stroke),
                  }))}
                  note="Development data (M3), mean over runs"
                />
              ) : null
            }
          />
          {Object.entries(CONDITION_COLOR).map(([condition, color], i) => (
            <Line
              key={condition}
              type="monotone"
              dataKey={condition}
              stroke={color}
              strokeWidth={1.9}
              dot={{ r: 2.4, fill: color, strokeWidth: 0 }}
              animationDuration={850}
              animationBegin={i * 120}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* resplit stability — 20 development resplits                             */
/* ---------------------------------------------------------------------- */

export function ResplitStabilityChart({ height = 132 }: { height?: number }) {
  const data = charts.development.resplitStability.map((row) => ({
    resplit: row.resplit,
    cleanAuroc: row.cleanAuroc,
    frozenPatternFolds: row.frozenPatternFolds,
  }))
  const mean =
    data.reduce((sum, row) => sum + row.cleanAuroc, 0) / Math.max(data.length, 1)

  return (
    <>
      <ChartDescription>
        Clean AUROC across the {data.length} development resplits, mean {mean.toFixed(4)}. Bars are
        shaded by how many of that resplit&rsquo;s five held-out folds picked the frozen pattern.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 10, bottom: 0, left: -12 }} barCategoryGap="18%">
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="resplit"
            tick={{ ...AXIS.tick, fontSize: 9 }}
            tickLine={false}
            axisLine={AXIS.line}
            interval={1}
          />
          <YAxis
            domain={[0.8, 0.845]}
            ticks={[0.81, 0.83]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
            width={42}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            cursor={{ fill: 'rgba(16,170,165,0.05)' }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={`Resplit ${payload[0].payload.resplit}`}
                  rows={[
                    {
                      name: 'Clean AUROC',
                      value: Number(payload[0].payload.cleanAuroc).toFixed(5),
                      color: CHART.clean,
                    },
                  ]}
                  note={`Frozen pattern picked on ${payload[0].payload.frozenPatternFolds}/5 held-out folds`}
                />
              ) : null
            }
          />
          <ReferenceLine y={mean} stroke={CHART.navy} strokeDasharray="4 4" strokeWidth={1} />
          <Bar dataKey="cleanAuroc" radius={[3, 3, 0, 0]} animationDuration={800}>
            {data.map((row) => (
              <Cell
                key={row.resplit}
                fill={row.frozenPatternFolds > 0 ? CHART.clean : '#B7CCD4'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* candidate ranking — top patterns by mean development excess NLL         */
/* ---------------------------------------------------------------------- */

export function CandidateRankingChart({
  count = 8,
  height = 190,
}: {
  count?: number
  height?: number
}) {
  const frozen = development.predeclared
  const data = charts.development.candidates.slice(0, count).map((row) => ({
    name: row.name,
    short: row.analytes.join('+'),
    excess: row.meanExcessNll,
    isFrozen: row.name === 'BUN+Glucose+Na',
  }))

  return (
    <>
      <ChartDescription>
        Top {count} of {charts.development.nCandidates} candidate patterns by mean development
        excess NLL. The frozen pattern was chosen by 1-SE parsimony over {frozen.n_resplits}{' '}
        resplits, not by taking the maximum.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 2, right: 14, bottom: 2, left: 4 }}
          barCategoryGap="24%"
        >
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 0.016]}
            ticks={[0, 0.005, 0.01, 0.015]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
            tickFormatter={(v: number) => v.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="short"
            tick={{ ...AXIS.tick, fontSize: 9.5 }}
            tickLine={false}
            axisLine={false}
            width={132}
          />
          <Tooltip
            cursor={{ fill: 'rgba(16,170,165,0.05)' }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={String(payload[0].payload.name)}
                  rows={[
                    {
                      name: 'Mean excess NLL',
                      value: Number(payload[0].value).toFixed(6),
                      color: payload[0].payload.isFrozen ? CHART.clean : CHART.control,
                    },
                  ]}
                  note={
                    payload[0].payload.isFrozen
                      ? 'Frozen pattern — selected by 1-SE parsimony'
                      : 'Development only; never confirmed on the holdout'
                  }
                />
              ) : null
            }
          />
          <Bar dataKey="excess" radius={[0, 4, 4, 0]} animationDuration={850} maxBarSize={15}>
            {data.map((row) => (
              <Cell key={row.name} fill={row.isFrozen ? CHART.clean : '#BFD3DA'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* M2 representation grid — why values_mask was frozen                     */
/* ---------------------------------------------------------------------- */

export function RepresentationChart({ height = 158 }: { height?: number }) {
  const rows = charts.representations
    .filter((row) => row.model === 'xgboost')
    .map((row) => ({
      name: row.representation.replace(/_/g, ' '),
      auroc: row.auroc,
      isFrozen: row.representation === 'values_mask',
    }))

  return (
    <>
      <ChartDescription>
        M2 executed representation grid, XGBoost. values_mask is the representation carried into the
        frozen pipeline.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 8, right: 10, bottom: 2, left: -10 }} barCategoryGap="22%">
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ ...AXIS.tick, fontSize: 8.5 }}
            tickLine={false}
            axisLine={AXIS.line}
            interval={0}
            angle={-14}
            textAnchor="end"
            height={44}
          />
          <YAxis
            domain={[0.5, 0.88]}
            ticks={[0.5, 0.7, 0.85]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
            width={42}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            cursor={{ fill: 'rgba(16,170,165,0.05)' }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={String(payload[0].payload.name)}
                  rows={[
                    {
                      name: 'AUROC (XGBoost)',
                      value: Number(payload[0].value).toFixed(6),
                      color: CHART.clean,
                    },
                  ]}
                  note={payload[0].payload.isFrozen ? 'Frozen representation' : 'M2 development'}
                />
              ) : null
            }
          />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]} animationDuration={800} maxBarSize={34}>
            {rows.map((row) => (
              <Cell key={row.name} fill={row.isFrozen ? CHART.clean : '#BFD3DA'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* effect size: predeclared threshold vs development vs confirmed          */
/* ---------------------------------------------------------------------- */

export function EffectSizeChart({ height = 186 }: { height?: number }) {
  const data = [
    {
      name: 'Predeclared MDE',
      value: development.detectability.minimum_detectable_effect,
      kind: 'threshold',
    },
    {
      name: 'Development (out-of-selection)',
      value: development.detectability.out_of_selection_delta,
      kind: 'development',
    },
    {
      name: 'Confirmed 95% lower bound',
      value: confirmation.lowerBound,
      kind: 'confirmed',
    },
    { name: 'Confirmed Δ C', value: confirmation.deltaC, kind: 'confirmed' },
  ]

  const FILL: Record<string, string> = {
    threshold: '#B7CCD4',
    development: CHART.control,
    confirmed: CHART.clean,
  }

  return (
    <>
      <ChartDescription>
        The confirmed excess NLL of {confirmation.deltaC.toFixed(6)} exceeds the development
        out-of-selection estimate and is more than twice the predeclared minimum detectable effect.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 16, bottom: 4, left: 4 }}
          barCategoryGap="26%"
        >
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 0.02]}
            ticks={[0, 0.005, 0.01, 0.015, 0.02]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
            tickFormatter={(v: number) => v.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ ...AXIS.tick, fontSize: 9.5 }}
            tickLine={false}
            axisLine={false}
            width={150}
          />
          <Tooltip
            cursor={{ fill: 'rgba(16,170,165,0.05)' }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={String(payload[0].payload.name)}
                  rows={[
                    {
                      name: 'Excess NLL',
                      value: `+${Number(payload[0].value).toFixed(6)}`,
                      color: FILL[String(payload[0].payload.kind)],
                    },
                  ]}
                  note={
                    payload[0].payload.kind === 'confirmed'
                      ? 'Set-c holdout, executed once'
                      : 'Development data only'
                  }
                />
              ) : null
            }
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} animationDuration={850} maxBarSize={18}>
            {data.map((row) => (
              <Cell key={row.name} fill={FILL[row.kind]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* compact sparkline used in Overview snapshot cards                       */
/* ---------------------------------------------------------------------- */

export function Sparkline({
  data,
  dataKey,
  color = CHART.clean,
  height = 58,
  domain,
}: {
  data: readonly Record<string, number>[]
  dataKey: string
  color?: string
  height?: number
  domain?: [number, number]
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data as Record<string, number>[]} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
        <YAxis hide domain={domain ?? ['auto', 'auto']} />
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={1.7}
          dot={false}
          animationDuration={900}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
