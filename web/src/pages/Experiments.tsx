import { useState } from 'react'
import {
  CheckCircle2,
  CircleSlash,
  FlaskConical,
  GitCommitHorizontal,
  Info,
  Lock,
  Repeat2,
  Search,
  Target,
} from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import CliniverseOrb from '../components/orb/CliniverseOrb'
import { Card, CardHead } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Progress from '../components/ui/Progress'
import Tooltip from '../components/ui/Tooltip'
import CopyButton from '../components/ui/CopyButton'
import { EffectSizeChart } from '../components/charts/LabCharts'
import { useOrbSize } from '../hooks/useOrbSize'
import {
  formatCount,
  confirmation,
  development,
  gates,
  milestones,
  type Milestone,
  type MilestoneState,
} from '../data/cliniverseResults'
import { repoFileUrl } from '../data/evidenceLinks'
import './experiments.css'

const STATE_TONE: Record<MilestoneState, 'green' | 'teal' | 'amber' | 'grey'> = {
  confirmed: 'green',
  complete: 'teal',
  closed: 'amber',
  'not-started': 'grey',
}

const FILTERS = ['All milestones', 'Confirmed', 'Development', 'Closed'] as const
type Filter = (typeof FILTERS)[number]

function matches(milestone: Milestone, filter: Filter): boolean {
  if (filter === 'All milestones') return true
  if (filter === 'Confirmed') return milestone.state === 'confirmed'
  if (filter === 'Closed') return milestone.state === 'closed'
  return milestone.state === 'complete'
}

export default function Experiments() {
  const orbSize = useOrbSize(330)
  const [filter, setFilter] = useState<Filter>('All milestones')
  const [selectedId, setSelectedId] = useState(milestones[milestones.length - 1].id)
  const [query, setQuery] = useState('')

  const visible = milestones.filter(
    (milestone) =>
      matches(milestone, filter) &&
      (query === '' ||
        `${milestone.id} ${milestone.name} ${milestone.verdict}`
          .toLowerCase()
          .includes(query.toLowerCase())),
  )
  const selected = milestones.find((m) => m.id === selectedId) ?? milestones[0]

  return (
    <PageTransition title="Experiments">
      <header className="cv-page-head">
        <h1 className="cv-page-title">Experiments</h1>
        <p>The executed milestone history — replay and inspect, not a run queue.</p>
        <div className="cv-page-actions">
          <Button
            variant="primary"
            icon={<FlaskConical size={14} />}
            disabledReason="No experiment runner is exposed to the browser. Milestones are executed from the repository, and the set-c holdout is spent — see docs/M5_V2_SETC_CONFIRMATION.md §9."
          >
            New experiment
          </Button>
        </div>
      </header>

      <div className="cv-note cv-note--amber ex-banner">
        <Lock size={15} strokeWidth={2} aria-hidden />
        <span>
          <strong>This page is a record, not a control surface.</strong> Every entry below was
          executed from the repository and is reproducible from its committed artifacts. Nothing
          here launches a scientific pipeline, and the one-shot holdout cannot be run again.
        </span>
      </div>

      {/* -------------------- orchestration graph -------------------- */}
      <div className="ex-hero">
        <Card index={0} hover={false} className="ex-graph" tone="dark">
          <div className="ex-graph-head">
            <Badge tone="mint">Execution history</Badge>
            <span className="ex-graph-title">Milestone graph</span>
            <span className="cv-meta">
              {milestones.length} executed milestones · one confirmed holdout test
            </span>
          </div>

          <div className="ex-graph-body">
            <ul className="ex-graph-legend">
              <LegendRow icon={<CheckCircle2 size={13} />} label="Complete" value={milestones.filter((m) => m.state === 'complete').length} tone="teal" />
              <LegendRow icon={<Target size={13} />} label="Confirmed" value={milestones.filter((m) => m.state === 'confirmed').length} tone="green" />
              <LegendRow icon={<CircleSlash size={13} />} label="Closed" value={milestones.filter((m) => m.state === 'closed').length} tone="amber" />
              <LegendRow icon={<Repeat2 size={13} />} label="Resplits" value={development.predeclared.n_resplits} tone="teal" />
            </ul>

            <div className="ex-orb">
              <CliniverseOrb
                variant="experiment"
                size={orbSize}
                intensity={0.55}
                label="Milestone orchestration graph"
                caption={
                  <>
                    <span className="orb-caption-line">{milestones.length} milestones</span>
                    <span className="orb-caption-line">1 confirmed</span>
                  </>
                }
              />
            </div>

            <ul className="ex-graph-nodes">
              {milestones.slice(-4).map((milestone) => (
                <li key={milestone.id} className={milestone.state === 'confirmed' ? 'is-final' : ''}>
                  <span className="ex-node-dot" aria-hidden />
                  <span>
                    <strong>{milestone.id}</strong>
                    <em>{milestone.verdict}</em>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Card>

        <Card index={1} hover={false} className="ex-inspector">
          <CardHead
            title={selected.name}
            sub={selected.id}
            action={<Badge tone={STATE_TONE[selected.state]}>{selected.verdict}</Badge>}
          />
          <p className="cv-body ex-summary">{selected.summary}</p>

          {selected.headline && (
            <div className="ex-headline">
              {selected.headline.map((item) => (
                <div key={item.label}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          )}

          <dl className="ex-meta">
            {selected.nPatients && (
              <Row k="Patients" v={formatCount(selected.nPatients)} />
            )}
            <Row k="Report" v={selected.report} mono />
            {selected.gitSha && <Row k="Commit" v={selected.gitSha.slice(0, 12)} mono copy={selected.gitSha} />}
          </dl>

          <Button
            variant="secondary"
            icon={<GitCommitHorizontal size={14} />}
            href={repoFileUrl(selected.report)}
            external
            title={`Open ${selected.report} in the repository`}
          >
            Open report
          </Button>
        </Card>
      </div>

      {/* -------------------- table + comparison --------------------- */}
      <div className="ex-lower">
        <Card index={2} hover={false}>
          <CardHead
            title="Milestone record"
            sub={`${visible.length} of ${milestones.length} shown`}
            action={
              <div className="ex-controls">
                <label className="ex-search">
                  <Search size={13} strokeWidth={2} aria-hidden />
                  <input
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Filter milestones…"
                    aria-label="Filter milestones"
                  />
                </label>
                <div className="ex-filters" role="tablist">
                  {FILTERS.map((item) => (
                    <button
                      key={item}
                      type="button"
                      role="tab"
                      aria-selected={filter === item}
                      className={`ex-filter${filter === item ? ' is-active' : ''}`}
                      onClick={() => setFilter(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            }
          />

          {visible.length === 0 ? (
            <div className="cv-empty">
              <Search size={26} strokeWidth={1.5} />
              <strong>No milestone matches</strong>
              <p>Clear the filter or search term to see the full record.</p>
            </div>
          ) : (
            <div className="cv-table-scroll">
              <table className="cv-table">
                <thead>
                  <tr>
                    <th>Milestone</th>
                    <th>Name</th>
                    <th>Verdict</th>
                    <th>Patients</th>
                    <th>Commit</th>
                    <th>Report</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((milestone) => (
                    <tr
                      key={milestone.id}
                      className={milestone.id === selectedId ? 'is-selected' : ''}
                      onClick={() => setSelectedId(milestone.id)}
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          setSelectedId(milestone.id)
                        }
                      }}
                    >
                      <td>
                        <strong>{milestone.id}</strong>
                      </td>
                      <td>{milestone.name}</td>
                      <td>
                        <Badge tone={STATE_TONE[milestone.state]}>{milestone.verdict}</Badge>
                      </td>
                      <td>{milestone.nPatients ? formatCount(milestone.nPatients) : '—'}</td>
                      <td className="cv-mono">{milestone.gitSha?.slice(0, 7) ?? '—'}</td>
                      <td className="ex-report">{milestone.report.replace('docs/', '')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card index={3}>
          <CardHead
            title="Effect size"
            sub="Predeclared threshold vs development vs confirmed"
          />
          <EffectSizeChart height={186} />
          <p className="cv-meta ex-note">
            The confirmed effect exceeded both the predeclared minimum detectable effect and the
            development out-of-selection estimate. The detectability gate was not optimistic.
          </p>
        </Card>
      </div>

      {/* -------------------- reproducibility ------------------------ */}
      <div className="ex-stats">
        <Card index={4}>
          <CardHead
            icon={<Repeat2 size={16} strokeWidth={1.8} />}
            title="Selection stability"
            sub="Gate G2"
          />
          <div className="ex-stat">
            <strong>
              {development.gates.G2_majority_stability.count}/
              {development.gates.G2_majority_stability.of}
            </strong>
            <span>resplits selected the frozen pattern</span>
          </div>
          <Progress
            value={development.gates.G2_majority_stability.pi}
            tone="teal"
            height={6}
            label="Selection frequency"
          />
          <p className="cv-meta ex-note">
            Exactly the majority threshold of{' '}
            {development.gates.G2_majority_stability.required} — the gate passed by a single
            resplit, so the exact membership is provisional.
          </p>
        </Card>

        <Card index={5}>
          <CardHead
            icon={<Target size={16} strokeWidth={1.8} />}
            title="Shrinkage"
            sub="After repairing held-out-fold leakage"
          />
          <div className="ex-stat">
            <strong>
              {(development.estimates.shrinkage_fraction * 100).toFixed(1)}%
            </strong>
            <span>under the v2 procedure</span>
          </div>
          <Progress
            value={development.estimates.shrinkage_fraction}
            tone="orange"
            height={6}
            label="Shrinkage fraction"
          />
          <p className="cv-meta ex-note">
            Against 58% under v1&rsquo;s non-comparable naive procedure. The exact percentages are
            not a like-for-like estimate of bias reduction.
          </p>
        </Card>

        <Card index={6}>
          <CardHead
            icon={<CheckCircle2 size={16} strokeWidth={1.8} />}
            title="Development gates"
            sub="All four required to freeze"
          />
          <ul className="ex-gates">
            {gates.map((gate) => (
              <li key={gate.id}>
                <CheckCircle2 size={12} strokeWidth={2.4} aria-hidden />
                <b>{gate.id}</b>
                <span>{gate.name}</span>
                <Badge tone="green">pass</Badge>
              </li>
            ))}
          </ul>
        </Card>

        <Card index={7}>
          <CardHead
            icon={<Info size={16} strokeWidth={1.8} />}
            title="Holdout execution"
            sub="One test, one decision"
          />
          <dl className="ex-meta">
            <Row k="Cohort" v={`set-c · ${formatCount(confirmation.cohort.n_patients)}`} />
            <Row k="Bootstrap" v={formatCount(confirmation.nBootstrap)} />
            <Row k="Seed" v={String(confirmation.bootstrapSeed)} />
            <Row k="Alternatives run" v="0" />
          </dl>
          <div className="cv-note cv-note--navy ex-spent">
            <Lock size={13} strokeWidth={2} aria-hidden />
            <span>
              <Tooltip term="set-c">Set C</Tooltip> is spent. No further set-c experiment is
              authorised by this result.
            </span>
          </div>
        </Card>
      </div>
    </PageTransition>
  )
}

/* ---------------------------------------------------------------------- */

function LegendRow({
  icon,
  label,
  value,
  tone,
}: {
  icon: React.ReactNode
  label: string
  value: number
  tone: string
}) {
  return (
    <li className={`ex-legend ex-legend--${tone}`}>
      <span className="ex-legend-icon" aria-hidden>
        {icon}
      </span>
      <span className="ex-legend-label">{label}</span>
      <b>{value}</b>
    </li>
  )
}

function Row({
  k,
  v,
  mono,
  copy,
}: {
  k: string
  v: string
  mono?: boolean
  copy?: string
}) {
  return (
    <div className="cv-kv">
      <dt>{k}</dt>
      <dd className={mono ? 'cv-mono' : undefined}>
        {v}
        {copy && <CopyButton value={copy} label={k} />}
      </dd>
    </div>
  )
}
