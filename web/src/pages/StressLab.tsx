import { useMemo, useState } from 'react'
import {
  Activity,
  CheckCircle2,
  FlaskConical,
  Info,
  Layers,
  Lock,
  RotateCcw,
  Search,
  Sparkles,
  TriangleAlert,
} from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import CliniverseOrb from '../components/orb/CliniverseOrb'
import { useOrbSize } from '../hooks/useOrbSize'
import { Card, CardHead } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Progress from '../components/ui/Progress'
import Tooltip from '../components/ui/Tooltip'
import Timeline from '../components/ui/Timeline'
import { SeverityResponseChart, ResplitStabilityChart } from '../components/charts/LabCharts'
import { CHART, Legend } from '../components/charts/chartKit'
import {
  formatCount,
  charts,
  confirmation,
  development,
  gates,
  investigationTimeline,
} from '../data/cliniverseResults'
import './stresslab.css'

/**
 * The severity slider replays the executed M3 severity sweep. It cannot run a
 * new evaluation — Set C is spent and the frozen contract forbids a second
 * look — so every control here is labelled as a replay of committed results.
 */
const SEVERITIES = [0, 0.25, 0.5, 0.75] as const

const FROZEN = 'BUN+Glucose+Na'

export default function StressLab() {
  const orbSize = useOrbSize(286)
  const [severity, setSeverity] = useState<number>(0.5)
  const [condition, setCondition] = useState('group_structured')
  const [selected, setSelected] = useState(FROZEN)

  const series = charts.stressResponse.series
  const cleanRow = series.find((s) => s.condition === 'none')

  const active = useMemo(() => {
    if (severity === 0) return cleanRow
    return series.find((s) => s.condition === condition && s.severity === severity) ?? cleanRow
  }, [series, condition, severity, cleanRow])

  const selectedCandidate =
    charts.development.candidates.find((c) => c.name === selected) ??
    charts.development.candidates[0]

  const severityLookup: Record<
    string,
    { requestedSeverity: number; realizedSeverityMean: number; totalRemovedCells: number }
  > = charts.stressResponse.severityReport

  const severityReport =
    severityLookup[`${severity}|${condition}`] ?? severityLookup['0.0|none']

  const intensity = severity === 0 ? 0.12 : 0.28 + severity * 0.85

  return (
    <PageTransition title="Stress Lab">
      <header className="cv-page-head">
        <h1 className="cv-page-title">Stress Lab</h1>
        <Badge tone="amber" icon={<RotateCcw size={11} strokeWidth={2.2} />}>
          Replay of committed results
        </Badge>
        <p>
          Explore the executed withholding experiments interactively. No control here runs a new
          scientific evaluation.
        </p>
      </header>

      <div className="cv-note cv-note--amber sl-banner">
        <Lock size={15} strokeWidth={2} aria-hidden />
        <span>
          <strong>Set C is spent.</strong> The single pre-registered holdout test has been consumed
          and no further set-c experiment is authorised. Everything below replays results already
          committed to the repository — changing a control re-reads saved artifacts, it does not
          re-evaluate the model, and no combination shown here becomes a confirmed finding by being
          selected.
        </span>
      </div>

      <div className="sl-main">
        {/* ------------------ scenario builder ---------------------- */}
        <Card index={0} hover={false} className="sl-builder">
          <CardHead
            title="Scenario replay"
            sub="Select a committed M3 condition"
            action={
              <Button
                variant="ghost"
                icon={<RotateCcw size={13} />}
                onClick={() => {
                  setSeverity(0.5)
                  setCondition('group_structured')
                }}
              >
                Reset
              </Button>
            }
          />

          <fieldset className="sl-field">
            <legend>
              Withholding condition
              <Tooltip text="The three removal mechanisms executed in M3. Cell-random and variable-matched are the controls that separate 'which analytes' from 'how much data'." />
            </legend>
            <div className="sl-radios">
              {[
                { id: 'group_structured', label: 'Group-structured' },
                { id: 'cell_random', label: 'Cell-random control' },
                { id: 'variable_matched_scattered', label: 'Variable-matched control' },
              ].map((option) => (
                <label
                  key={option.id}
                  className={`sl-radio${condition === option.id ? ' is-active' : ''}`}
                >
                  <input
                    type="radio"
                    name="condition"
                    value={option.id}
                    checked={condition === option.id}
                    onChange={() => setCondition(option.id)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="sl-field">
            <legend>
              Requested severity
              <Tooltip text="The fraction of eligible cells the mechanism was asked to remove. Realised severity differs because removal is clipped to naturally observed availability." />
            </legend>
            <div className="sl-severity">
              {SEVERITIES.map((value) => (
                <button
                  key={value}
                  type="button"
                  className={`sl-sev${severity === value ? ' is-active' : ''}`}
                  onClick={() => setSeverity(value)}
                  aria-pressed={severity === value}
                >
                  {value === 0 ? 'Clean' : value.toFixed(2)}
                </button>
              ))}
            </div>
            <div className="sl-sev-readout">
              <span>Realised severity</span>
              <b>{severityReport ? severityReport.realizedSeverityMean.toFixed(4) : '—'}</b>
            </div>
            <div className="sl-sev-readout">
              <span>Cells removed</span>
              <b>{severityReport ? formatCount(severityReport.totalRemovedCells) : '—'}</b>
            </div>
          </fieldset>

          <div className="sl-engine-readout">
            <span className="cv-meta">Engine intensity</span>
            <Progress value={intensity} tone={severity > 0.5 ? 'orange' : 'teal'} height={6} />
            <span className="sl-engine-value">{intensity.toFixed(2)}</span>
          </div>

          <div className="sl-library">
            <div className="sl-library-head">
              <span className="cv-section-title">Candidate library</span>
              <Badge tone="grey">{development.nCandidates}</Badge>
            </div>
            <ul className="sl-candidates">
              {charts.development.candidates.slice(0, 6).map((candidate) => (
                <li key={candidate.name}>
                  <button
                    type="button"
                    className={`sl-candidate${selected === candidate.name ? ' is-active' : ''}`}
                    onClick={() => setSelected(candidate.name)}
                    aria-pressed={selected === candidate.name}
                    title={candidate.analytes.join(' + ')}
                  >
                    <span className="sl-candidate-icon" aria-hidden>
                      <Layers size={13} strokeWidth={1.9} />
                    </span>
                    <span className="sl-candidate-text">
                      <strong>{candidate.analytes.join(' + ')}</strong>
                      <em>excess NLL {candidate.meanExcessNll.toFixed(6)}</em>
                    </span>
                    {candidate.name === FROZEN ? (
                      <Badge tone="teal">frozen</Badge>
                    ) : (
                      <span className="sl-candidate-val">
                        {candidate.oosFoldPicks}/{charts.development.oosFoldPickTotal}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </Card>

        {/* ------------------ engine -------------------------------- */}
        <Card index={1} hover={false} className="sl-engine" tone="dark">
          <div className="sl-engine-head">
            <Badge tone="mint">Replaying committed run</Badge>
            <span className="sl-engine-title">Stress Engine</span>
            <span className="cv-meta">
              M3 severity sweep · 8,000 development patients
            </span>
          </div>

          <div className="sl-orb-wrap">
            <CliniverseOrb
              variant="stress"
              size={orbSize}
              intensity={intensity}
              label={`Stress engine at replayed intensity ${intensity.toFixed(2)}`}
              caption={
                <>
                  <strong>Realised severity</strong>
                  <b>{severityReport ? severityReport.realizedSeverityMean.toFixed(3) : '—'}</b>
                  <em>{condition.replace(/_/g, ' ')}</em>
                </>
              }
            />
          </div>

          <div className="sl-engine-metrics">
            <EngineMetric label="AUROC" value={active ? active.auroc.toFixed(4) : '—'} />
            <EngineMetric label="NLL" value={active ? active.nll.toFixed(4) : '—'} />
            <EngineMetric
              label="Calib. intercept"
              value={active ? active.calibrationIntercept.toFixed(4) : '—'}
              warn={active ? active.calibrationIntercept > 0.2 : false}
            />
            <EngineMetric
              label="Mean risk"
              value={active ? active.meanPredictedRisk.toFixed(4) : '—'}
            />
          </div>
        </Card>

        {/* ------------------ insights ------------------------------ */}
        <Card index={2} hover={false} className="sl-insights">
          <CardHead
            icon={<CheckCircle2 size={16} strokeWidth={1.9} />}
            title="Confirmed pattern"
            sub="The one result that reached the holdout"
            action={<Badge tone="green">Confirmed</Badge>}
          />

          <div className="sl-confirm-metrics">
            <div className="sl-cm">
              <span>
                <Tooltip term="Delta_C">Δ&#8202;C</Tooltip>
              </span>
              <strong>+{confirmation.deltaC.toFixed(6)}</strong>
            </div>
            <div className="sl-cm">
              <span>95% lower bound</span>
              <strong className="is-good">+{confirmation.lowerBound.toFixed(6)}</strong>
            </div>
            <div className="sl-cm">
              <span>AUROC drop</span>
              <strong>{confirmation.aurocDrop.toFixed(6)}</strong>
            </div>
          </div>

          <div className="sl-pattern">
            <span className="cv-meta">Frozen pattern</span>
            <div className="sl-analytes">
              {confirmation.pattern.map((analyte) => (
                <span className="sl-analyte" key={analyte}>
                  {analyte}
                </span>
              ))}
            </div>
            <p className="cv-meta">
              Selected in {development.gates.G2_majority_stability.count}/
              {development.gates.G2_majority_stability.of} development resplits — exactly the
              majority threshold, so the exact membership is provisional.
            </p>
          </div>

          <div className="sl-gates">
            <span className="cv-meta">Development gates</span>
            {gates.map((gate) => (
              <div className="sl-gate" key={gate.id}>
                <CheckCircle2 size={12} strokeWidth={2.4} aria-hidden />
                <b>{gate.id}</b>
                <span>{gate.name}</span>
                <Badge tone="green">pass</Badge>
              </div>
            ))}
          </div>

          <div className="sl-detect">
            <div className="sl-detect-row">
              <span>Out-of-selection estimate</span>
              <b>+{development.detectability.out_of_selection_delta.toFixed(6)}</b>
            </div>
            <div className="sl-detect-row">
              <span>Minimum detectable effect</span>
              <b>+{development.detectability.minimum_detectable_effect.toFixed(6)}</b>
            </div>
            <Progress
              value={
                development.detectability.minimum_detectable_effect /
                development.detectability.out_of_selection_delta
              }
              tone="teal"
              height={5}
              label="Minimum detectable effect against the out-of-selection estimate"
            />
          </div>
        </Card>
      </div>

      {/* ------------------ lower analysis row ---------------------- */}
      <div className="sl-lower">
        <Card index={3}>
          <CardHead
            icon={<Activity size={16} strokeWidth={1.8} />}
            title="Performance vs severity"
            sub="M3 executed sweep, uncalibrated"
            action={
              <Legend
                items={[
                  { label: 'Group-structured', color: CHART.withheld },
                  { label: 'Cell-random', color: CHART.control },
                  { label: 'Variable-matched', color: CHART.clean },
                ]}
              />
            }
          />
          <SeverityResponseChart metric="auroc" height={150} />
          <p className="cv-meta sl-note">
            Discrimination falls modestly under every condition — which is exactly why AUROC alone
            does not surface this failure.
          </p>
        </Card>

        <Card index={4}>
          <CardHead
            icon={<TriangleAlert size={16} strokeWidth={1.8} />}
            title="Calibration drift vs severity"
            sub="The signature ordinary monitoring misses"
          />
          <SeverityResponseChart metric="calibrationIntercept" height={150} />
          <p className="cv-meta sl-note">
            Group-structured removal drifts the intercept far faster than amount-matched controls at
            the same severity.
          </p>
        </Card>

        <Card index={5}>
          <CardHead
            icon={<Search size={16} strokeWidth={1.8} />}
            title="Resplit stability"
            sub={`Clean AUROC across ${charts.development.resplitStability.length} resplits`}
          />
          <ResplitStabilityChart height={132} />
          <p className="cv-meta sl-note">
            Teal bars are resplits where at least one held-out fold selected the frozen pattern in
            the out-of-selection procedure.
          </p>
        </Card>

        <Card index={6} className="sl-inspector">
          <CardHead
            icon={<FlaskConical size={16} strokeWidth={1.8} />}
            title="Candidate inspector"
            sub="Development estimates only"
          />
          <div className="sl-insp-name">
            {selectedCandidate.analytes.map((analyte) => (
              <span className="sl-analyte" key={analyte}>
                {analyte}
              </span>
            ))}
          </div>
          <dl className="sl-insp-list">
            <Row k="Mean excess NLL" v={selectedCandidate.meanExcessNll.toFixed(6)} />
            <Row k="Mean AUROC" v={selectedCandidate.meanAuroc.toFixed(6)} />
            <Row k="Mean AUROC drop" v={selectedCandidate.meanAurocDrop.toFixed(6)} />
            <Row
              k="Out-of-selection picks"
              v={`${selectedCandidate.oosFoldPicks} / ${charts.development.oosFoldPickTotal}`}
            />
          </dl>
          {selectedCandidate.name === FROZEN ? (
            <div className="cv-note cv-note--teal">
              <Sparkles size={13} strokeWidth={2} aria-hidden />
              <span>
                This is the frozen pattern and the only one evaluated on the holdout.
              </span>
            </div>
          ) : (
            <div className="cv-note cv-note--navy">
              <Info size={13} strokeWidth={2} aria-hidden />
              <span>
                Development estimate only. This candidate was never evaluated on set C and cannot
                be.
              </span>
            </div>
          )}
        </Card>
      </div>

      <Card index={7} hover={false}>
        <CardHead title="Investigation timeline" sub="From exhaustive search to a spent holdout" />
        <Timeline steps={investigationTimeline} />
      </Card>
    </PageTransition>
  )
}

function EngineMetric({
  label,
  value,
  warn,
}: {
  label: string
  value: string
  warn?: boolean
}) {
  return (
    <div className={`sl-em${warn ? ' is-warn' : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="cv-kv">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  )
}
