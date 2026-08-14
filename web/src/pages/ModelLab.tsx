import { useState } from 'react'
import {
  Boxes,
  CalendarClock,
  CheckCircle2,
  Clock3,
  Database,
  FileCode2,
  Gauge,
  GitBranch,
  Lock,
  Percent,
  ShieldCheck,
  Sigma,
  Snowflake,
  Table2,
} from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import CliniverseOrb from '../components/orb/CliniverseOrb'
import { useOrbSize } from '../hooks/useOrbSize'
import { Card, CardHead, LiveBadge } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Progress from '../components/ui/Progress'
import Tooltip from '../components/ui/Tooltip'
import CopyButton from '../components/ui/CopyButton'
import { RepresentationChart } from '../components/charts/LabCharts'
import {
  formatCount,
  charts,
  cleanVsWithheld,
  confirmation,
  development,
  freeze,
  frozenCalibrator,
  gates,
  product,
  setc,
} from '../data/cliniverseResults'
import './modellab.css'

const TABS = ['Representation grid', 'Training data', 'Feature space', 'Guardrails', 'Lineage'] as const
type Tab = (typeof TABS)[number]

export default function ModelLab() {
  const orbSize = useOrbSize(252)
  const [tab, setTab] = useState<Tab>('Representation grid')

  const split = freeze.split
  const totalLabelled = split.n_total + confirmation.cohort.n_patients

  return (
    <PageTransition title="Model Lab">
      <header className="cv-page-head">
        <h1 className="cv-page-title">Model Lab</h1>
        <LiveBadge label="Frozen" />
        <p>The frozen pipeline under test, its inputs, outputs and freeze governance.</p>
      </header>

      {/* -------------------- model identity strip -------------------- */}
      <Card index={0} hover={false} className="ml-identity">
        <div className="ml-identity-main">
          <span className="ml-identity-icon" aria-hidden>
            <Snowflake size={22} strokeWidth={1.7} />
          </span>
          <div className="ml-identity-text">
            <span className="cv-meta">Frozen model</span>
            <h2>
              {freeze.model.estimator}
              <Badge tone="navy">{freeze.provenance.git_sha.slice(0, 7)}</Badge>
            </h2>
            <span className="ml-identity-tags">
              <Badge tone="teal">{freeze.preprocessing.representation}</Badge>
              <Badge tone="grey">Tabular · {freeze.preprocessing.cutoff_hours}h cutoff</Badge>
              <Badge tone="grey">In-hospital mortality</Badge>
            </span>
          </div>
        </div>

        <dl className="ml-identity-meta">
          <div>
            <dt>State</dt>
            <dd className="ml-frozen">
              <Lock size={12} strokeWidth={2.2} aria-hidden /> Frozen · contract sealed
            </dd>
          </div>
          <div>
            <dt>Fitted on</dt>
            <dd>{formatCount(split.n_final_train)} clean training rows</dd>
          </div>
          <div>
            <dt>Provenance</dt>
            <dd>
              {freeze.provenance.git_dirty ? 'dirty' : 'clean tree'} · {freeze.provenance.python}
            </dd>
          </div>
        </dl>

        <Button
          variant="secondary"
          icon={<FileCode2 size={14} />}
          disabledReason="The freeze record is a committed repository artifact; open final_freeze.json in the repository."
        >
          View freeze record
        </Button>
      </Card>

      {/* -------------------- inputs · engine · outputs ---------------- */}
      <div className="ml-engine">
        <Card index={1} className="ml-io">
          <CardHead title="Model inputs" sub="What the frozen pipeline consumes" />
          <IoRow
            icon={<Table2 size={15} strokeWidth={1.8} />}
            label="Features"
            value={`${freeze.preprocessing.n_features} built · ${freeze.model.n_features_used} used`}
            foot="values_mask representation"
          />
          <IoRow
            icon={<Database size={15} strokeWidth={1.8} />}
            label="Development cohort"
            value={`${formatCount(split.n_total)} patients`}
            foot={`sets ${freeze.provenance.sets.join(' + ')} · set-c excluded`}
          />
          <IoRow
            icon={<Clock3 size={15} strokeWidth={1.8} />}
            label="Observation window"
            value={`${freeze.preprocessing.cutoff_hours} hours`}
            foot={`of a ${product.dataset.horizonHours}h record`}
          />
          <IoRow
            icon={<Percent size={15} strokeWidth={1.8} />}
            label="Grid missingness"
            value={`${(product.dataset.missingness * 100).toFixed(2)}%`}
            foot="binned hourly occupancy 20.25%"
          />
          <IoRow
            icon={<Sigma size={15} strokeWidth={1.8} />}
            label="Imputation"
            value={freeze.preprocessing.imputation_strategy}
            foot="fitted once on clean training rows"
          />
        </Card>

        <Card index={2} hover={false} className="ml-orb-card">
          <div className="ml-orb-head">
            <span className="cv-card-title">Model heartbeat</span>
            <span className="cv-meta">Deserialised and applied as-is — nothing is refitted</span>
          </div>
          <div className="ml-orb-flow">
            <span className="ml-flow-label ml-flow-in">
              Inputs
              <i aria-hidden />
            </span>
            <CliniverseOrb
              variant="model"
              size={orbSize}
              intensity={0.3}
              label="Frozen model inference engine, showing a steady heartbeat"
              caption={
                <>
                  <strong>Frozen pipeline</strong>
                  <b>{freeze.model.n_features_used}</b>
                  <em>features used</em>
                </>
              }
            />
            <span className="ml-flow-label ml-flow-out">
              <i aria-hidden />
              Predictions
            </span>
          </div>
          <div className="ml-orb-foot">
            <span>
              <b>{freeze.model.hyperparameters.n_estimators}</b> trees
            </span>
            <span>
              depth <b>{freeze.model.hyperparameters.max_depth}</b>
            </span>
            <span>
              lr <b>{freeze.model.hyperparameters.learning_rate}</b>
            </span>
            <span>
              seed <b>{freeze.model.random_state}</b>
            </span>
          </div>
        </Card>

        <Card index={3} className="ml-io">
          <CardHead title="Model outputs" sub="On the set-c holdout, clean condition" />
          <IoRow
            icon={<Gauge size={15} strokeWidth={1.8} />}
            label="Mean predicted risk"
            value={cleanVsWithheld.meanRisk.clean.toFixed(5)}
            foot={`against prevalence ${confirmation.cohort.prevalence.toFixed(5)}`}
          />
          <IoRow
            icon={<Sigma size={15} strokeWidth={1.8} />}
            label="Calibrator"
            value={`Platt · slope ${frozenCalibrator.slope.toFixed(5)}`}
            foot={`intercept ${frozenCalibrator.intercept.toFixed(5)} — never refitted under withholding`}
          />
          <IoRow
            icon={<Boxes size={15} strokeWidth={1.8} />}
            label="Probability reliability"
            value={`NLL ${cleanVsWithheld.nll.clean.toFixed(5)}`}
            foot={`Brier ${cleanVsWithheld.brier.clean.toFixed(5)}`}
          />
          <IoRow
            icon={<CheckCircle2 size={15} strokeWidth={1.8} />}
            label="Discrimination"
            value={confirmation.cleanAuroc.toFixed(6)}
            foot={`AUPRC ${cleanVsWithheld.auprc.clean.toFixed(5)}`}
          />
          <IoRow
            icon={<CalendarClock size={15} strokeWidth={1.8} />}
            label="Freeze diagnostic"
            value={freeze.fittingDiagnostics.mean_raw_calibration_prediction.toFixed(6)}
            foot="reproduced exactly at pre-flight before the unlock"
          />
        </Card>

        <Card index={4} className="ml-gov">
          <CardHead
            title="Freeze &amp; audit status"
            sub="Development gates and execution integrity"
            action={<Badge tone="green">All passed</Badge>}
          />
          <ul className="ml-gates">
            {gates.map((gate) => (
              <li key={gate.id}>
                <span className="ml-gate-icon" aria-hidden>
                  <CheckCircle2 size={13} strokeWidth={2.2} />
                </span>
                <span className="ml-gate-body">
                  <strong>
                    {gate.id} · {gate.name}
                  </strong>
                  <em>{gate.detail}</em>
                </span>
              </li>
            ))}
          </ul>

          <div className="ml-integrity">
            <span className="cv-meta">Execution integrity</span>
            <IntegrityRow
              label="Set C loaded during freeze"
              value={freeze.setCAccess.loaded_during_freeze ? 'yes' : 'no'}
              good={!freeze.setCAccess.loaded_during_freeze}
            />
            <IntegrityRow
              label="Artifacts hash-verified pre-flight"
              value="3 / 3"
              good
            />
            <IntegrityRow
              label="Alternative analyses run"
              value={String(setc.alternativeAnalysesRun.count)}
              good={setc.alternativeAnalysesRun.count === 0}
            />
          </div>
        </Card>
      </div>

      {/* -------------------- performance strip ------------------------ */}
      <Card index={5} hover={false}>
        <CardHead
          title="Core model performance"
          sub="Clean condition, set-c holdout (n = 4,000) — one evaluation, never repeated"
          status="Confirmed"
          action={
            <span className="cv-meta">
              executed {confirmation.executedAt.slice(0, 10)}
            </span>
          }
        />
        <div className="ml-perf">
          <Perf label="AUROC" value={confirmation.cleanAuroc.toFixed(6)} term="AUROC" />
          <Perf label="AUPRC" value={cleanVsWithheld.auprc.clean.toFixed(6)} />
          <Perf label="Brier" value={cleanVsWithheld.brier.clean.toFixed(6)} term="Brier" />
          <Perf label="NLL" value={cleanVsWithheld.nll.clean.toFixed(6)} term="NLL" />
          <Perf
            label="Calibration intercept"
            value={cleanVsWithheld.calibrationIntercept.clean.toFixed(6)}
            term="calibration intercept"
          />
          <Perf
            label="Calibration slope"
            value={cleanVsWithheld.calibrationSlope.clean.toFixed(6)}
            term="calibration slope"
          />
          <Perf label="Patients" value={formatCount(confirmation.cohort.n_patients)} />
        </div>
      </Card>

      {/* -------------------- lower area ------------------------------- */}
      <div className="ml-lower">
        <Card index={6} hover={false}>
          <div className="cv-tabs" role="tablist">
            {TABS.map((item) => (
              <button
                key={item}
                type="button"
                role="tab"
                aria-selected={tab === item}
                className={`cv-tab${tab === item ? ' is-active' : ''}`}
                onClick={() => setTab(item)}
              >
                {item}
                {tab === item && <span className="cv-tab-underline" />}
              </button>
            ))}
          </div>

          {tab === 'Representation grid' && (
            <div className="ml-tabbody">
              <div className="ml-chart">
                <span className="cv-section-title">M2 representation grid — XGBoost</span>
                <RepresentationChart height={172} />
              </div>
              <div className="cv-table-scroll">
                <table className="cv-table">
                  <thead>
                    <tr>
                      <th>Representation</th>
                      <th>Model</th>
                      <th>AUROC</th>
                      <th>Brier</th>
                      <th>NLL</th>
                      <th>Features</th>
                    </tr>
                  </thead>
                  <tbody>
                    {charts.representations
                      .filter((row) => row.model === 'xgboost')
                      .slice(0, 5)
                      .map((row) => (
                        <tr
                          key={row.runId}
                          className={row.representation === 'values_mask' ? 'is-selected' : ''}
                        >
                          <td>
                            {row.representation}
                            {row.representation === 'values_mask' && (
                              <Badge tone="teal">frozen</Badge>
                            )}
                          </td>
                          <td>{row.model}</td>
                          <td>{row.auroc?.toFixed(6)}</td>
                          <td>{row.brier?.toFixed(6)}</td>
                          <td>{row.nll?.toFixed(6)}</td>
                          <td>{row.nFeatures}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <p className="cv-meta ml-tabnote">
                M2 development grid on 8,000 A+B patients. These are not holdout numbers — set C was
                never loaded during M2.
              </p>
            </div>
          )}

          {tab === 'Training data' && (
            <div className="ml-tabbody ml-split">
              <SplitBar
                label="Final training"
                n={split.n_final_train}
                total={totalLabelled}
                foot={`prevalence ${split.train_prevalence}`}
                tone="teal"
              />
              <SplitBar
                label="Final calibration"
                n={split.n_final_calibration}
                total={totalLabelled}
                foot={`prevalence ${split.calibration_prevalence}`}
                tone="navy"
              />
              <SplitBar
                label="Set-C holdout"
                n={confirmation.cohort.n_patients}
                total={totalLabelled}
                foot={`prevalence ${confirmation.cohort.prevalence} · spent`}
                tone="orange"
              />
              <p className="cv-meta ml-tabnote">
                Stratified by {split.stratified_by}, seed {split.seed}, disjoint:{' '}
                {String(split.disjoint)}. {formatCount(totalLabelled)} labelled patients in
                total.
              </p>
            </div>
          )}

          {tab === 'Feature space' && (
            <div className="ml-tabbody">
              <dl className="ml-dl">
                <Kv k="Features built" v={String(freeze.preprocessing.n_features)} />
                <Kv k="Features used by the model" v={String(freeze.model.n_features_used)} />
                <Kv k="Representation" v={freeze.preprocessing.representation} />
                <Kv k="Imputation" v={freeze.preprocessing.imputation_strategy} />
                <Kv k="Imputer seed" v={String(freeze.preprocessing.imputer_seed)} />
                <Kv k="Catalogue version" v={freeze.provenance.catalogue_version} />
                <Kv
                  k="Eligible control pool"
                  v={`${freeze.contract.eligible_control_pool_n} variables`}
                />
              </dl>
              <p className="cv-meta ml-tabnote">
                Co-measurement groups are reconstructed clusters (<code>*_like</code>), never
                verified laboratory orders.
              </p>
            </div>
          )}

          {tab === 'Guardrails' && (
            <div className="ml-tabbody">
              <ul className="ml-guards">
                {(freeze.contract.forbidden as string[]).map((rule) => (
                  <li key={rule}>
                    <ShieldCheck size={13} strokeWidth={2} aria-hidden />
                    {rule}
                  </li>
                ))}
              </ul>
              <p className="cv-meta ml-tabnote">
                The five prohibitions written into the frozen set-c contract before the holdout was
                opened.
              </p>
            </div>
          )}

          {tab === 'Lineage' && (
            <div className="ml-tabbody">
              <ol className="ml-lineage">
                <LineageStep
                  sha={development.provenance.git_sha}
                  title="M5-v2 development"
                  body={`${development.nCandidates} candidates over ${development.predeclared.n_resplits} resplits · verdict v2-STABLE`}
                />
                <LineageStep
                  sha={freeze.provenance.git_sha}
                  title="Final model freeze"
                  body={`${formatCount(split.n_final_train)} / ${formatCount(split.n_final_calibration)} isolation · contract sealed · set C not loaded during freeze`}
                />
                <LineageStep
                  sha={confirmation.executedAt.slice(0, 10)}
                  title="One-shot set-c confirmation"
                  body="Three artifacts hash-verified, pipeline deserialised, exactly one analysis run"
                  last
                />
              </ol>
            </div>
          )}
        </Card>

        <Card index={7}>
          <CardHead
            icon={<GitBranch size={16} strokeWidth={1.8} />}
            title="Frozen artifacts"
            sub="Hash-verified before the holdout was unlocked"
          />
          <ul className="ml-artifacts">
            {Object.entries(freeze.artifactHashes).map(([name, hash]) => (
              <li key={name}>
                <div className="ml-artifact-head">
                  <strong>{name}</strong>
                  <CopyButton value={hash} label={`${name} hash`} />
                </div>
                <code className="cv-hash">{hash}</code>
              </li>
            ))}
          </ul>
          <div className="cv-note cv-note--navy ml-artifact-note">
            <ShieldCheck size={14} strokeWidth={2} aria-hidden />
            <span>
              The run aborts on any hash mismatch. Every pre-flight step used development data only.
            </span>
          </div>
        </Card>
      </div>
    </PageTransition>
  )
}

/* ---------------------------------------------------------------------- */

function IoRow({
  icon,
  label,
  value,
  foot,
}: {
  icon: React.ReactNode
  label: string
  value: string
  foot?: string
}) {
  return (
    <div className="ml-io-row">
      <span className="ml-io-icon" aria-hidden>
        {icon}
      </span>
      <span className="ml-io-text">
        <em>{label}</em>
        <strong>{value}</strong>
        {foot && <span>{foot}</span>}
      </span>
    </div>
  )
}

function IntegrityRow({
  label,
  value,
  good,
}: {
  label: string
  value: string
  good: boolean
}) {
  return (
    <div className="ml-integrity-row">
      <span>{label}</span>
      <b className={good ? 'is-good' : ''}>{value}</b>
    </div>
  )
}

function Perf({ label, value, term }: { label: string; value: string; term?: string }) {
  return (
    <div className="ml-perf-cell">
      <span className="cv-meta">
        {term ? <Tooltip term={term}>{label}</Tooltip> : label}
      </span>
      <strong>{value}</strong>
    </div>
  )
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div className="cv-kv">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  )
}

function SplitBar({
  label,
  n,
  total,
  foot,
  tone,
}: {
  label: string
  n: number
  total: number
  foot: string
  tone: 'teal' | 'navy' | 'orange'
}) {
  return (
    <div className="ml-splitbar">
      <div className="ml-splitbar-head">
        <strong>{label}</strong>
        <span>
          {formatCount(n)} <em>({((n / total) * 100).toFixed(1)}%)</em>
        </span>
      </div>
      <Progress value={n / total} tone={tone} height={7} label={`${label} ${n}`} />
      <span className="cv-meta">{foot}</span>
    </div>
  )
}

function LineageStep({
  sha,
  title,
  body,
  last,
}: {
  sha: string
  title: string
  body: string
  last?: boolean
}) {
  return (
    <li className={`ml-lineage-step${last ? ' is-last' : ''}`}>
      <span className="ml-lineage-dot" aria-hidden />
      <div>
        <strong>{title}</strong>
        <code className="cv-mono">{sha.slice(0, 12)}</code>
        <em>{body}</em>
      </div>
    </li>
  )
}
