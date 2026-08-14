import { Link } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  Box,
  CheckCircle2,
  LineChart,
  Target,
  TriangleAlert,
} from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import CliniverseOrb, { type OrbNode } from '../components/orb/CliniverseOrb'
import { Card, CardHead, LiveBadge } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Timeline from '../components/ui/Timeline'
import Progress from '../components/ui/Progress'
import Tooltip from '../components/ui/Tooltip'
import { CountUp } from '../components/ui/Metric'
import { Sparkline } from '../components/charts/LabCharts'
import { useOrbSize } from '../hooks/useOrbSize'
import {
  AnalyteDropletIcon,
  AnalyteIonIcon,
  AnalyteTubeIcon,
} from '../components/icons/AnalyteIcons'
import { CHART } from '../components/charts/chartKit'
import {
  formatCount,
  charts,
  cleanVsWithheld,
  confirmation,
  development,
  freeze,
  investigationTimeline,
} from '../data/cliniverseResults'
import './overview.css'

/**
 * Chip placement follows the locked reference: BUN above the core, Glucose
 * lower-left, Na lower-right. Widths are deliberately unequal.
 */
const NODES: readonly OrbNode[] = [
  {
    id: 'bun',
    label: 'BUN',
    sub: 'Blood Urea Nitrogen',
    icon: <AnalyteTubeIcon />,
    x: 0.63,
    y: 0.06,
    width: 186,
  },
  {
    id: 'glucose',
    label: 'Glucose',
    sub: 'Serum',
    icon: <AnalyteDropletIcon />,
    x: 0.12,
    y: 0.79,
    width: 148,
  },
  { id: 'na', label: 'Na⁺', sub: 'Sodium', icon: <AnalyteIonIcon />, x: 0.86, y: 0.84, width: 118 },
]

/** Development resplit AUROC, used as the Model Lab snapshot trend. */
const resplitTrend = charts.development.resplitStability.map((row) => ({
  x: row.resplit,
  cleanAuroc: row.cleanAuroc,
}))

const topCandidates = charts.development.candidates
  .filter((row) => row.oosFoldPicks > 0)
  .sort((a, b) => b.oosFoldPicks - a.oosFoldPicks)
  .slice(0, 3)

export default function Overview() {
  const orbSize = useOrbSize()
  const riskDropPct =
    (cleanVsWithheld.meanRisk.withheld / cleanVsWithheld.meanRisk.clean - 1) * 100

  return (
    <PageTransition title="Overview">
      <div className="ov-hero">
        {/* ---------------- hero: narrative + orb --------------------- */}
        <Card index={0} hover={false} className="ov-hero-main">
          <div className="ov-hero-text">
            <span className="cv-eyebrow ov-eyebrow">
              <i className="cv-live-dot" aria-hidden />
              Active vulnerability
            </span>
            <h1 className="cv-page-title ov-title">{confirmation.patternLabel}</h1>
            <p className="cv-body ov-lede">
              A three-analyte pattern, selected entirely on development data, that degrades this
              frozen model&rsquo;s probability reliability on a quarantined cohort — while
              discrimination barely moves.
            </p>
            <Link to="/stress-lab">
              <Button variant="secondary" trailing={<ArrowRight size={15} />}>
                Explore in Stress Lab
              </Button>
            </Link>
          </div>

          <div className="ov-hero-orb">
            <CliniverseOrb
              variant="overview"
              size={orbSize}
              nodes={NODES}
              intensity={0.5}
              label={`Cliniverse investigation orb showing the confirmed failure pattern ${confirmation.patternLabel}`}
              caption={
                <>
                  <span className="orb-caption-line">Confirmed</span>
                  <span className="orb-caption-line">Failure Pattern</span>
                </>
              }
            />
          </div>
        </Card>

        {/* ---------------- confirmation panel ------------------------ */}
        <Card index={1} hover={false} className="ov-confirm">
          <header className="ov-confirm-head">
            <span className="ov-confirm-icon" aria-hidden>
              <CheckCircle2 size={17} strokeWidth={2} />
            </span>
            <h2 className="cv-card-title">Discrimination-silent failure confirmed</h2>
            <Badge tone="green">Both conditions passed</Badge>
          </header>

          <div className="ov-metrics">
            <div className="ov-metric">
              <span className="ov-metric-label">
                <Tooltip term="Delta_C">Δ&#8202;C (excess NLL)</Tooltip>
              </span>
              <strong className="ov-metric-value">
                +<CountUp value={confirmation.deltaC} places={4} />
              </strong>
              <span className="ov-metric-foot ov-up">Higher is worse</span>
            </div>
            <div className="ov-metric">
              <span className="ov-metric-label">
                <Tooltip term="one-sided 95% lower bound">95% lower bound</Tooltip>
              </span>
              <strong className="ov-metric-value">
                +<CountUp value={confirmation.lowerBound} places={4} />
              </strong>
              <span className="ov-metric-foot ov-good">
                <CheckCircle2 size={11} strokeWidth={2.4} /> Bound exceeds zero
              </span>
            </div>
            <div className="ov-metric">
              <span className="ov-metric-label">AUROC drop</span>
              <strong className="ov-metric-value">
                <CountUp value={confirmation.aurocDrop} places={4} />
              </strong>
              <span className="ov-metric-foot">
                ≤ {confirmation.aurocDropCeiling} ceiling
              </span>
            </div>
          </div>

          <div className="cv-note cv-note--teal ov-explain">
            <Target size={14} strokeWidth={2} aria-hidden />
            <span>
              <strong>Effect detected without a change in discrimination.</strong> Probability
              reliability degrades while ranking is largely preserved.
            </span>
          </div>

          {/* the descriptive diagnostics behind that sentence */}
          <div className="ov-secondary">
            <span className="cv-meta ov-secondary-head">
              Secondary diagnostics — clean → withheld
              <Tooltip text="Reported for interpretation only. These are not part of the frozen decision rule." />
            </span>
            <div className="ov-secondary-grid">
              <Shift label="NLL" clean={cleanVsWithheld.nll.clean} withheld={cleanVsWithheld.nll.withheld} />
              <Shift
                label="Brier"
                clean={cleanVsWithheld.brier.clean}
                withheld={cleanVsWithheld.brier.withheld}
              />
              <Shift
                label="Calib. intercept"
                clean={cleanVsWithheld.calibrationIntercept.clean}
                withheld={cleanVsWithheld.calibrationIntercept.withheld}
              />
              <Shift
                label="Mean risk"
                clean={cleanVsWithheld.meanRisk.clean}
                withheld={cleanVsWithheld.meanRisk.withheld}
              />
            </div>
          </div>

          <footer className="ov-confirm-foot">
            <span>Confirmed on</span>
            <b>{formatCount(confirmation.cohort.n_patients)} quarantined patients</b>
            <i aria-hidden />
            <b>{formatCount(confirmation.nBootstrap)} bootstrap resamples</b>
            <i aria-hidden />
            <b>{freeze.contract.control_repeats} amount-matched controls</b>
          </footer>
        </Card>
      </div>

      {/* ---------------- three snapshots ---------------------------- */}
      <div className="ov-snapshots">
        <Card index={2}>
          <CardHead
            icon={<Box size={16} strokeWidth={1.8} />}
            title="Model Lab"
            sub="Frozen model, clean holdout performance"
            status="Frozen"
            action={
              <Link to="/model-lab">
                <Button variant="ghost" trailing={<ArrowRight size={14} />}>
                  View model
                </Button>
              </Link>
            }
          />
          <div className="ov-snap-metrics">
            <Snap label="AUROC" value={confirmation.cleanAuroc.toFixed(4)} />
            <Snap label="NLL" value={cleanVsWithheld.nll.clean.toFixed(4)} />
            <Snap
              label="Mean risk"
              value={`${(cleanVsWithheld.meanRisk.clean * 100).toFixed(1)}%`}
            />
            <Snap label="Patients" value={formatCount(confirmation.cohort.n_patients)} />
          </div>
          <div className="ov-snap-chart">
            <Sparkline data={resplitTrend} dataKey="cleanAuroc" color={CHART.clean} height={62} />
          </div>
          <p className="cv-meta">
            Clean AUROC across {resplitTrend.length} development resplits
          </p>
        </Card>

        <Card index={3}>
          <CardHead
            icon={<Activity size={16} strokeWidth={1.8} />}
            title="Stress Lab"
            sub="Stability-aware candidate search"
            status="Development"
            action={
              <Link to="/stress-lab">
                <Button variant="ghost" trailing={<ArrowRight size={14} />}>
                  Open lab
                </Button>
              </Link>
            }
          />
          <div className="ov-stress">
            <div className="ov-stress-count">
              <strong>{development.nCandidates}</strong>
              <span>candidates enumerated</span>
            </div>
            <div className="ov-stress-list">
              <span className="cv-meta ov-stress-caption">
                Most frequent out-of-selection picks
                <Tooltip text="Across 20 resplits × 5 held-out folds = 100 nested selections. This is not the headline 11/20 resplit majority." />
              </span>
              {topCandidates.map((row, i) => (
                <div className="ov-rank" key={row.name}>
                  <span className="ov-rank-n">{i + 1}</span>
                  <span className="ov-rank-name">{row.name.replace(/\+/g, ' + ')}</span>
                  <span className="ov-rank-bar">
                    <Progress
                      value={row.oosFoldPicks / charts.development.oosFoldPickTotal}
                      height={4}
                      label={`${row.oosFoldPicks} of ${charts.development.oosFoldPickTotal}`}
                    />
                  </span>
                  <span className="ov-rank-val">
                    {row.oosFoldPicks}/{charts.development.oosFoldPickTotal}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        <Card index={4}>
          <CardHead
            icon={<LineChart size={16} strokeWidth={1.8} />}
            title="Reliability Report"
            sub="Clean vs withheld on the holdout"
            status="Confirmed"
            action={
              <Link to="/reliability-report">
                <Button variant="ghost" trailing={<ArrowRight size={14} />}>
                  View report
                </Button>
              </Link>
            }
          />
          <div className="ov-compare">
            <div className="ov-compare-col">
              <span className="cv-meta">AUROC</span>
              <div className="ov-compare-row">
                <b>{confirmation.cleanAuroc.toFixed(4)}</b>
                <ArrowRight size={13} className="ov-arrow" aria-hidden />
                <b className="ov-warn">{confirmation.withheldAuroc.toFixed(4)}</b>
              </div>
              <span className="cv-meta ov-sub">
                barely moves — {confirmation.aurocDrop.toFixed(4)}
              </span>
            </div>
            <div className="ov-compare-col">
              <span className="cv-meta">Mean predicted risk</span>
              <div className="ov-compare-row">
                <b>{cleanVsWithheld.meanRisk.clean.toFixed(4)}</b>
                <ArrowRight size={13} className="ov-arrow" aria-hidden />
                <b className="ov-warn">{cleanVsWithheld.meanRisk.withheld.toFixed(4)}</b>
              </div>
              <span className="cv-meta ov-sub ov-warn">{riskDropPct.toFixed(1)}%</span>
            </div>
          </div>
          <div className="cv-note cv-note--amber ov-alert">
            <TriangleAlert size={14} strokeWidth={2} aria-hidden />
            <span>
              Systematic risk under-prediction against a set-c prevalence of{' '}
              {confirmation.cohort.prevalence.toFixed(5)}.
            </span>
          </div>
        </Card>
      </div>

      {/* ---------------- timeline ---------------------------------- */}
      <Card index={5} hover={false}>
        <CardHead
          title="Investigation timeline"
          sub="How the confirmed pattern was found, frozen and tested"
          action={<LiveBadge label="Set C spent" />}
        />
        <Timeline steps={investigationTimeline} />
      </Card>
    </PageTransition>
  )
}

function Shift({
  label,
  clean,
  withheld,
}: {
  label: string
  clean: number
  withheld: number
}) {
  return (
    <div className="ov-shift">
      <span>{label}</span>
      <b>
        {clean.toFixed(4)}
        <i aria-hidden>→</i>
        <em>{withheld.toFixed(4)}</em>
      </b>
    </div>
  )
}

function Snap({ label, value }: { label: string; value: string }) {
  return (
    <div className="ov-snap">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
