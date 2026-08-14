import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileText,
  Quote,
  ShieldAlert,
  ShieldCheck,
  Table2,
} from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import CliniverseOrb from '../components/orb/CliniverseOrb'
import { useOrbSize } from '../hooks/useOrbSize'
import { Card, CardHead } from '../components/ui/Card'
import { Badge, EvidenceBadge } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Tooltip from '../components/ui/Tooltip'
import Timeline from '../components/ui/Timeline'
import CopyButton from '../components/ui/CopyButton'
import {
  MeanRiskShiftChart,
  ReliabilityChart,
  RiskDistributionChart,
  RocChart,
} from '../components/charts/ReportCharts'
import { CHART, Legend } from '../components/charts/chartKit'
import {
  formatCount,
  charts,
  cleanVsWithheld,
  confirmation,
  development,
  freeze,
  historicalDisclosure,
  investigationTimeline,
  keyFindings,
  limitations,
  narrativeInsight,
  provenanceBlemish,
  setc,
} from '../data/cliniverseResults'
import './report.css'

const LEGEND = [
  { label: 'Clean', color: CHART.clean },
  { label: 'Withheld', color: CHART.withheld },
]

export default function ReliabilityReport() {
  const orbSize = useOrbSize(168)
  return (
    <PageTransition title="Reliability Report">
      {/* -------------------- report header ------------------------- */}
      <Card index={0} hover={false} className="rr-head">
        <div className="rr-head-text">
          <span className="cv-eyebrow">Reliability report</span>
          <h1 className="cv-page-title rr-title">{confirmation.patternLabel}</h1>
          <p className="cv-body rr-lede">
            Clean versus failure-inducing comparison on the quarantined set-c holdout, executed once
            against a contract frozen before the cohort was opened.
          </p>
          <dl className="rr-meta">
            <Meta k="Model" v={`${freeze.model.estimator} · ${freeze.preprocessing.representation}`} />
            <Meta k="Outcome" v="In-hospital mortality" />
            <Meta k="Executed" v={confirmation.executedAt.slice(0, 10)} />
            <Meta k="Freeze" v={freeze.provenance.git_sha.slice(0, 7)} />
          </dl>
        </div>

        <div className="rr-head-orb">
          <CliniverseOrb
            variant="report"
            size={orbSize}
            intensity={0.22}
            label="Reliability report emblem"
          />
        </div>

        <div className="rr-verdict">
          <div className="rr-verdict-row">
            <span className="rr-verdict-icon" aria-hidden>
              <CheckCircle2 size={16} strokeWidth={2.1} />
            </span>
            <span>Discrimination-silent failure</span>
            <Badge tone="green">CONFIRMED</Badge>
          </div>
          <div className="rr-verdict-row">
            <span className="cv-meta">Primary rule</span>
            <b>{confirmation.rule}</b>
          </div>
          <div className="rr-verdict-row">
            <span className="cv-meta">One-sided 95% lower bound</span>
            <b className="is-good">+{confirmation.lowerBound.toFixed(6)}</b>
          </div>
          <div className="rr-verdict-row">
            <span className="cv-meta">Analyses executed</span>
            <b>{setc.alternativeAnalysesRun.count + 1} — no alternatives</b>
          </div>
        </div>

        <div className="rr-head-actions">
          <Button variant="secondary" icon={<FileText size={14} />} disabledReason="The written report is a committed repository file: docs/M5_V2_SETC_CONFIRMATION.md">
            Written report
          </Button>
          <Button variant="dark" icon={<Download size={14} />} disabledReason="Result artifacts are committed to the repository under experiments/robustness/results/m5v2_setc/">
            Export data
          </Button>
        </div>
      </Card>

      {/* -------------------- comparison strip ---------------------- */}
      <Card index={1} hover={false}>
        <CardHead
          title="Clean vs withheld comparison"
          sub={`Set-c holdout · n = ${formatCount(confirmation.cohort.n_patients)} · ${confirmation.cohort.n_deaths} deaths · prevalence ${confirmation.cohort.prevalence}`}
          action={<EvidenceBadge evidence="confirmed" />}
        />
        <div className="rr-compare">
          <Compare
            label="Discrimination (AUROC)"
            term="AUROC"
            clean={confirmation.cleanAuroc}
            withheld={confirmation.withheldAuroc}
            places={4}
            verdict="Small — below the 0.02 ceiling"
            tone="ok"
          />
          <Compare
            label="Probability reliability (NLL)"
            term="NLL"
            clean={cleanVsWithheld.nll.clean}
            withheld={cleanVsWithheld.nll.withheld}
            places={5}
            verdict="Degraded"
            tone="warn"
          />
          <Compare
            label="Brier score"
            term="Brier"
            clean={cleanVsWithheld.brier.clean}
            withheld={cleanVsWithheld.brier.withheld}
            places={5}
            verdict="Degraded"
            tone="warn"
          />
          <Compare
            label="Mean predicted risk"
            clean={cleanVsWithheld.meanRisk.clean}
            withheld={cleanVsWithheld.meanRisk.withheld}
            places={5}
            verdict={`vs prevalence ${confirmation.cohort.prevalence}`}
            tone="warn"
          />
          <Compare
            label="Calibration intercept"
            term="calibration intercept"
            clean={cleanVsWithheld.calibrationIntercept.clean}
            withheld={cleanVsWithheld.calibrationIntercept.withheld}
            places={5}
            verdict="Large upward drift"
            tone="warn"
          />
        </div>
      </Card>

      {/* -------------------- charts -------------------------------- */}
      <div className="rr-charts">
        <Card index={2}>
          <CardHead
            title="ROC — clean vs withheld"
            sub="Ranking is preserved"
            action={<Legend items={LEGEND} />}
          />
          <RocChart height={168} />
          <div className="rr-chart-foot">
            <span>
              Clean <b>{confirmation.cleanAuroc.toFixed(4)}</b>
            </span>
            <span>
              Withheld <b className="is-warn">{confirmation.withheldAuroc.toFixed(4)}</b>
            </span>
            <span>
              Δ <b>{confirmation.aurocDrop.toFixed(4)}</b>
            </span>
          </div>
        </Card>

        <Card index={3}>
          <CardHead title="Calibration (reliability curve)" sub="Ten equal-count bins" />
          <ReliabilityChart height={168} />
          <div className="rr-chart-foot">
            <span>
              Intercept <b>{cleanVsWithheld.calibrationIntercept.clean.toFixed(4)}</b>
            </span>
            <span>
              → <b className="is-warn">{cleanVsWithheld.calibrationIntercept.withheld.toFixed(4)}</b>
            </span>
          </div>
        </Card>

        <Card index={4}>
          <CardHead title="Risk distribution" sub="4,000 holdout patients" />
          <RiskDistributionChart height={168} />
          <div className="rr-chart-foot">
            <span>
              Mean <b>{cleanVsWithheld.meanRisk.clean.toFixed(4)}</b>
            </span>
            <span>
              → <b className="is-warn">{cleanVsWithheld.meanRisk.withheld.toFixed(4)}</b>
            </span>
          </div>
        </Card>

        <Card index={5}>
          <CardHead title="Mean risk shift" sub="Against the amount-matched control" />
          <MeanRiskShiftChart height={168} />
          <div className="rr-chart-foot">
            <span>
              Control moves only{' '}
              <b>
                {(
                  charts.setc.meanRiskShift[1].meanRisk - charts.setc.meanRiskShift[0].meanRisk
                ).toFixed(5)}
              </b>
            </span>
          </div>
        </Card>
      </div>

      {/* -------------------- findings row -------------------------- */}
      <div className="rr-findings">
        <Card index={6}>
          <CardHead title="Key findings" />
          <ul className="rr-list">
            {keyFindings.map((finding) => (
              <li key={finding.text}>
                <CheckCircle2 size={13} strokeWidth={2.2} aria-hidden />
                <span>
                  {finding.text}
                  <em>{finding.source}</em>
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card index={7} className="rr-narrative">
          <CardHead icon={<Quote size={15} strokeWidth={1.9} />} title="Narrative insight" />
          <blockquote>{narrativeInsight.text}</blockquote>
          <footer>
            <EvidenceBadge evidence={narrativeInsight.evidence} />
            <span className="cv-meta">{narrativeInsight.source}</span>
          </footer>
        </Card>

        <Card index={8}>
          <CardHead
            icon={<ShieldCheck size={15} strokeWidth={1.9} />}
            title="Holdout integrity"
            sub="Verified before the unlock"
          />
          <div className="rr-integrity">
            <Integrity k="Cohort" v={`set-c · ${formatCount(confirmation.cohort.n_patients)}`} />
            <Integrity k="Set C loaded during freeze" v={String(freeze.setCAccess.loaded_during_freeze)} />
            <Integrity k="Bootstrap resamples" v={formatCount(confirmation.nBootstrap)} />
            <Integrity k="Amount-matched controls" v={String(freeze.contract.control_repeats)} />
            <Integrity k="Alternative analyses" v={String(setc.alternativeAnalysesRun.count)} />
          </div>
          <div className="cv-note cv-note--navy rr-disclosure">
            <ShieldAlert size={13} strokeWidth={2} aria-hidden />
            <span>{historicalDisclosure.text}</span>
          </div>
        </Card>
      </div>

      {/* -------------------- limitations + provenance -------------- */}
      <div className="rr-bottom">
        <Card index={9}>
          <CardHead
            icon={<AlertTriangle size={15} strokeWidth={1.9} />}
            title="Limitations that survive confirmation"
            action={<EvidenceBadge evidence="limitation" />}
          />
          <ol className="rr-limits">
            {limitations.map((limit, i) => (
              <li key={limit.text}>
                <span className="rr-limit-n">{i + 1}</span>
                <span>
                  {limit.text}
                  <em>{limit.source}</em>
                </span>
              </li>
            ))}
          </ol>
        </Card>

        <Card index={10}>
          <CardHead
            icon={<Table2 size={15} strokeWidth={1.9} />}
            title="Traceability"
            sub="Every number is recomputable from raw predictions"
          />
          <dl className="rr-trace">
            <Trace k="Cohort fingerprint" v={setc.provenance.cohort_fingerprint} copy />
            <Trace k="Record-id hash" v={setc.recordIdsHash} copy />
            <Trace k="Predictions" v={setc.predictionsFile.sha256} copy />
            <Trace k="Bootstrap seed" v={String(confirmation.bootstrapSeed)} />
            <Trace k="Repo HEAD at execution" v={setc.provenance.git_sha} copy />
            <Trace
              k="Development resplits"
              v={`${development.predeclared.n_resplits} · ${development.nCandidates} candidates`}
            />
          </dl>
          <div className="cv-note cv-note--amber rr-blemish">
            <AlertTriangle size={13} strokeWidth={2} aria-hidden />
            <span>{provenanceBlemish.text}</span>
          </div>
        </Card>
      </div>

      <Card index={11} hover={false}>
        <CardHead title="Investigation timeline" />
        <Timeline steps={investigationTimeline} />
      </Card>
    </PageTransition>
  )
}

/* ---------------------------------------------------------------------- */

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div className="rr-meta-item">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  )
}

function Compare({
  label,
  term,
  clean,
  withheld,
  places,
  verdict,
  tone,
}: {
  label: string
  term?: string
  clean: number
  withheld: number
  places: number
  verdict: string
  tone: 'ok' | 'warn'
}) {
  const delta = withheld - clean
  return (
    <div className="rr-cmp">
      <span className="rr-cmp-label">
        {term ? <Tooltip term={term}>{label}</Tooltip> : label}
      </span>
      <div className="rr-cmp-values">
        <b>{clean.toFixed(places)}</b>
        <i aria-hidden>→</i>
        <b className="is-withheld">{withheld.toFixed(places)}</b>
      </div>
      <div className="rr-cmp-foot">
        <span className={`rr-delta rr-delta--${tone}`}>
          {delta > 0 ? '+' : '−'}
          {Math.abs(delta).toFixed(places)}
        </span>
        <em>{verdict}</em>
      </div>
    </div>
  )
}

function Integrity({ k, v }: { k: string; v: string }) {
  return (
    <div className="cv-kv">
      <span className="cv-kv-k">{k}</span>
      <span className="cv-kv-v">{v}</span>
    </div>
  )
}

function Trace({ k, v, copy }: { k: string; v: string; copy?: boolean }) {
  return (
    <div className="rr-trace-row">
      <dt>{k}</dt>
      <dd>
        <code className="cv-hash">{copy ? `${v.slice(0, 24)}…` : v}</code>
        {copy && <CopyButton value={v} label={k} />}
      </dd>
    </div>
  )
}
