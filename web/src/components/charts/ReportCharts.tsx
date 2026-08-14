import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AXIS, CHART, ChartDescription, ChartTooltip } from './chartKit'
import { charts, cleanVsWithheld, confirmation } from '../../data/cliniverseResults'

const setc = charts.setc

/* ---------------------------------------------------------------------- */
/* ROC — clean vs withheld                                                 */
/* ---------------------------------------------------------------------- */

export function RocChart({ height = 168 }: { height?: number }) {
  const data = setc.roc.clean.map((point, i) => ({
    fpr: point.fpr,
    clean: point.tpr,
    withheld: setc.roc.withheld[i]?.tpr ?? null,
  }))

  return (
    <>
      <ChartDescription>
        ROC curves on the set-c holdout. Clean AUROC {confirmation.cleanAuroc.toFixed(6)}; withheld
        AUROC {confirmation.withheldAuroc.toFixed(6)}. The two curves nearly coincide — ranking is
        preserved.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: -8 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.5, 1]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={`False positive rate ${Number(label).toFixed(2)}`}
                  rows={payload.map((p) => ({
                    name: p.dataKey === 'clean' ? 'Clean' : 'Withheld',
                    value: Number(p.value).toFixed(3),
                    color: String(p.stroke),
                  }))}
                />
              ) : null
            }
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke={CHART.grid}
            strokeDasharray="4 4"
          />
          <Line
            type="monotone"
            dataKey="clean"
            stroke={CHART.clean}
            strokeWidth={2}
            dot={false}
            animationDuration={900}
          />
          <Line
            type="monotone"
            dataKey="withheld"
            stroke={CHART.withheld}
            strokeWidth={2}
            dot={false}
            animationDuration={900}
            animationBegin={150}
          />
        </LineChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* calibration / reliability curve                                         */
/* ---------------------------------------------------------------------- */

export function ReliabilityChart({ height = 168 }: { height?: number }) {
  const data = setc.reliability.clean.map((point, i) => ({
    predicted: point.meanPredicted,
    clean: point.observedRate,
    withheldPredicted: setc.reliability.withheld[i]?.meanPredicted ?? null,
    withheld: setc.reliability.withheld[i]?.observedRate ?? null,
  }))

  return (
    <>
      <ChartDescription>
        Reliability curve in ten equal-count bins. Under withholding the curve lifts above the
        diagonal: observed mortality exceeds predicted risk, so the model under-predicts.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: -8 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 0.6]}
            ticks={[0, 0.2, 0.4, 0.6]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
          />
          <YAxis
            domain={[0, 0.6]}
            ticks={[0, 0.3, 0.6]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title="Reliability bin"
                  rows={payload
                    .filter((p) => p.dataKey === 'clean' || p.dataKey === 'withheld')
                    .map((p) => ({
                      name: p.dataKey === 'clean' ? 'Clean observed' : 'Withheld observed',
                      value: Number(p.value).toFixed(3),
                      color: String(p.stroke),
                    }))}
                  note="Perfect calibration lies on the diagonal"
                />
              ) : null
            }
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 0.6, y: 0.6 },
            ]}
            stroke="#CBD9DF"
            strokeDasharray="4 4"
          />
          <Line
            type="monotone"
            dataKey="clean"
            stroke={CHART.clean}
            strokeWidth={2}
            dot={{ r: 2.4, fill: CHART.clean, strokeWidth: 0 }}
            animationDuration={900}
          />
          <Line
            type="monotone"
            dataKey="withheld"
            stroke={CHART.withheld}
            strokeWidth={2}
            dot={{ r: 2.4, fill: CHART.withheld, strokeWidth: 0 }}
            animationDuration={900}
            animationBegin={150}
          />
        </LineChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* predicted risk distribution                                             */
/* ---------------------------------------------------------------------- */

export function RiskDistributionChart({ height = 168 }: { height?: number }) {
  const data = setc.riskDistribution.clean.map((point, i) => ({
    risk: point.risk,
    clean: point.count,
    withheld: setc.riskDistribution.withheld[i]?.count ?? 0,
  }))

  return (
    <>
      <ChartDescription>
        Distribution of predicted risk across 4,000 holdout patients. Under withholding the mass
        shifts left: mean predicted risk falls from{' '}
        {cleanVsWithheld.meanRisk.clean.toFixed(4)} to {cleanVsWithheld.meanRisk.withheld.toFixed(4)}.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: 0 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="risk"
            type="number"
            domain={[0, 0.7]}
            ticks={[0, 0.25, 0.5, 0.75]}
            tickFormatter={(v: number) => v.toFixed(2)}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={AXIS.line}
          />
          <YAxis tick={AXIS.tick} tickLine={false} axisLine={false} width={46} />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={`Predicted risk ≈ ${Number(label).toFixed(3)}`}
                  rows={payload.map((p) => ({
                    name: p.dataKey === 'clean' ? 'Clean' : 'Withheld',
                    value: `${p.value} patients`,
                    color: String(p.color ?? p.stroke),
                  }))}
                />
              ) : null
            }
          />
          <Area
            type="monotone"
            dataKey="clean"
            stroke={CHART.clean}
            fill={CHART.cleanSoft}
            strokeWidth={1.8}
            animationDuration={900}
          />
          <Area
            type="monotone"
            dataKey="withheld"
            stroke={CHART.withheld}
            fill={CHART.withheldSoft}
            strokeWidth={1.8}
            animationDuration={900}
            animationBegin={150}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </>
  )
}

/* ---------------------------------------------------------------------- */
/* mean risk shift — clean vs amount-matched control vs withheld           */
/* ---------------------------------------------------------------------- */

export function MeanRiskShiftChart({ height = 168 }: { height?: number }) {
  const prevalence = confirmation.cohort.prevalence
  const data = setc.meanRiskShift.map((point) => ({
    stage: point.stage === 'Amount-matched control' ? 'Control' : point.stage,
    meanRisk: point.meanRisk,
    full: point.stage,
  }))

  return (
    <>
      <ChartDescription>
        Mean predicted risk under each condition against a set-c prevalence of{' '}
        {(prevalence * 100).toFixed(3)} percent. The amount-matched control barely moves; the
        withheld pattern drops the mean well below prevalence.
      </ChartDescription>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 10, right: 10, bottom: 2, left: 0 }} barCategoryGap="28%">
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="stage" tick={AXIS.tick} tickLine={false} axisLine={AXIS.line} />
          <YAxis
            domain={[0, 0.18]}
            ticks={[0, 0.05, 0.1, 0.15]}
            tick={AXIS.tick}
            tickLine={false}
            axisLine={false}
            width={46}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            cursor={{ fill: 'rgba(16,170,165,0.05)' }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  title={String(payload[0].payload.full)}
                  rows={[
                    {
                      name: 'Mean predicted risk',
                      value: Number(payload[0].value).toFixed(5),
                      color: CHART.clean,
                    },
                  ]}
                  note={`set-c prevalence ${prevalence.toFixed(5)}`}
                />
              ) : null
            }
          />
          <ReferenceLine
            y={prevalence}
            stroke={CHART.navy}
            strokeDasharray="4 4"
            strokeWidth={1.2}
            label={{
              value: `prevalence ${prevalence.toFixed(3)}`,
              position: 'insideTopRight',
              fontSize: 9.5,
              fill: CHART.navy,
            }}
          />
          <Bar dataKey="meanRisk" radius={[5, 5, 0, 0]} animationDuration={800} maxBarSize={54}>
            {data.map((entry) => (
              <Cell
                key={entry.stage}
                fill={
                  entry.stage === 'Clean'
                    ? CHART.clean
                    : entry.stage === 'Control'
                      ? CHART.control
                      : CHART.withheld
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  )
}
