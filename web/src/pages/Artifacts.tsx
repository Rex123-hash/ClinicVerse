import { useMemo, useState } from 'react'
import {
  Binary,
  Boxes,
  Database,
  Download,
  FileCode2,
  FileText,
  Image,
  Info,
  Search,
  Settings2,
  ShieldCheck,
  Sigma,
} from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import CliniverseOrb from '../components/orb/CliniverseOrb'
import { Card, CardHead } from '../components/ui/Card'
import { Badge, EvidenceBadge } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import CopyButton from '../components/ui/CopyButton'
import { useOrbSize } from '../hooks/useOrbSize'
import ConcentrationSurface from '../components/visuals/ConcentrationSurface'
import FailureSlice from '../components/visuals/FailureSlice'
import DamageLandscape from '../components/visuals/DamageLandscape'
import BurdenProfile from '../components/visuals/BurdenProfile'
import CaseExplorer from '../components/visuals/CaseExplorer'
import {
  artifacts,
  confirmation,
  formatCount,
  freeze,
  setc,
  visualArtifacts,
  type ArtifactKind,
  type VisualArtifact,
} from '../data/cliniverseResults'
import { repoFileUrl } from '../data/evidenceLinks'
import '../components/visuals/visuals.css'
import './artifacts.css'

const KIND_ICON: Record<ArtifactKind, React.ReactNode> = {
  result: <Sigma size={15} strokeWidth={1.8} />,
  predictions: <Binary size={15} strokeWidth={1.8} />,
  model: <Boxes size={15} strokeWidth={1.8} />,
  calibrator: <Settings2 size={15} strokeWidth={1.8} />,
  imputer: <Database size={15} strokeWidth={1.8} />,
  report: <FileText size={15} strokeWidth={1.8} />,
  figure: <Image size={15} strokeWidth={1.8} />,
  config: <FileCode2 size={15} strokeWidth={1.8} />,
}

const TABS = [
  { id: 'all', label: 'All' },
  { id: 'visual', label: 'Visual artifacts' },
  { id: 'tables', label: 'Tables & data' },
  { id: 'reports', label: 'Reports' },
  { id: 'downloads', label: 'Downloads' },
] as const
type TabId = (typeof TABS)[number]['id']

const TAB_KINDS: Partial<Record<TabId, readonly ArtifactKind[]>> = {
  tables: ['result', 'predictions', 'config'],
  reports: ['report'],
  downloads: ['model', 'calibrator', 'imputer', 'figure'],
}

/** Thumbnail preview for each visual artifact, rendered small inside its card. */
function Preview({ id }: { id: string }) {
  switch (id) {
    case 'concentration-surface':
      return <ConcentrationSurface width={300} height={150} compact />
    case 'failure-slice':
      return <FailureSlice height={150} compact />
    case 'damage-landscape':
      return <DamageLandscape height={150} count={6} compact />
    case 'burden-profile':
      return <BurdenProfile height={150} compact />
    case 'case-explorer':
      return <CaseExplorer compact />
    default:
      return null
  }
}

/** Full-size rendering for the inspector. */
function FullVisual({ id }: { id: string }) {
  switch (id) {
    case 'concentration-surface':
      return <ConcentrationSurface width={380} height={250} />
    case 'failure-slice':
      return <FailureSlice height={300} />
    case 'damage-landscape':
      return <DamageLandscape height={300} count={8} />
    case 'burden-profile':
      return <BurdenProfile height={280} />
    case 'case-explorer':
      return <CaseExplorer />
    default:
      return null
  }
}

export default function Artifacts() {
  const orbSize = useOrbSize(300)
  const [tab, setTab] = useState<TabId>('all')
  const [query, setQuery] = useState('')
  const [selectedVisual, setSelectedVisual] = useState<string>(visualArtifacts[0].id)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  const matches = (text: string) =>
    query === '' || text.toLowerCase().includes(query.toLowerCase())

  const visibleVisuals = useMemo(
    () =>
      tab === 'all' || tab === 'visual'
        ? visualArtifacts.filter((v) => matches(`${v.name} ${v.source} ${v.milestone}`))
        : [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tab, query],
  )

  const visibleFiles = useMemo(() => {
    if (tab === 'visual') return []
    const kinds = TAB_KINDS[tab]
    return artifacts.filter(
      (a) =>
        (!kinds || kinds.includes(a.kind)) &&
        matches(`${a.name} ${a.path} ${a.milestone}`),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, query])

  const visual = visualArtifacts.find((v) => v.id === selectedVisual) ?? visualArtifacts[0]
  const file = selectedFile ? artifacts.find((a) => a.id === selectedFile) : null

  return (
    <PageTransition title="Artifacts">
      <header className="cv-page-head">
        <h1 className="cv-page-title">Artifacts</h1>
        <p>Evidence, derived views and the provenance behind each one.</p>
        <div className="cv-page-actions">
          <label className="ar-search">
            <Search size={14} strokeWidth={2} aria-hidden />
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search artifacts, paths, milestones…"
              aria-label="Search artifacts"
            />
          </label>
          <Button
            variant="primary"
            disabledReason="Artifacts are produced by the experiment scripts in the repository and committed there; this application reads them, it does not create them."
          >
            New artifact
          </Button>
        </div>
      </header>

      {/* -------------------- evidence graph hero -------------------- */}
      <Card index={0} hover={false} className="ar-hero" tone="dark">
        <div className="ar-hero-text">
          <span className="ar-eyebrow">Evidence graph</span>
          <h2>Provenance of the confirmed result</h2>
          <p>
            Every number and every picture in this application resolves to one of these files. The
            three fitted objects were hash-verified before the holdout was unlocked, and the
            prediction vectors make each reported figure independently recomputable.
          </p>
          <div className="ar-hero-stats">
            <span>
              <b>{artifacts.length}</b> files
            </span>
            <i aria-hidden />
            <span>
              <b>{visualArtifacts.length}</b> derived views
            </span>
            <i aria-hidden />
            <span>
              <b>3</b> hash-sealed fitted objects
            </span>
            <i aria-hidden />
            <span>
              <b>{formatCount(confirmation.cohort.n_patients)}</b> holdout predictions
            </span>
          </div>
        </div>

        <div className="ar-hero-orb">
          <CliniverseOrb
            variant="artifact"
            size={orbSize}
            intensity={0.34}
            label="Provenance graph of the committed evidence"
            caption={
              <>
                <span className="orb-caption-line">Evidence</span>
                <span className="orb-caption-line">Provenance</span>
              </>
            }
          />
        </div>
      </Card>

      {/* -------------------- gallery + inspector -------------------- */}
      <div className="ar-body">
        <Card index={1} hover={false} className="ar-gallery">
          <div className="cv-tabs" role="tablist">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={tab === item.id}
                className={`cv-tab${tab === item.id ? ' is-active' : ''}`}
                onClick={() => setTab(item.id)}
              >
                {item.label}
                {tab === item.id && <span className="cv-tab-underline" />}
              </button>
            ))}
          </div>

          {visibleVisuals.length === 0 && visibleFiles.length === 0 && (
            <div className="cv-empty">
              <Search size={26} strokeWidth={1.5} />
              <strong>No artifact matches</strong>
              <p>Try a different tab or clear the search.</p>
            </div>
          )}

          {visibleVisuals.length > 0 && (
            <>
              {tab === 'all' && <span className="ar-section">Visual artifacts</span>}
              <ul className="ar-visual-grid">
                {visibleVisuals.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`ar-visual-card${
                        item.id === selectedVisual && !file ? ' is-selected' : ''
                      }`}
                      onClick={() => {
                        setSelectedVisual(item.id)
                        setSelectedFile(null)
                      }}
                      aria-pressed={item.id === selectedVisual && !file}
                    >
                      <span className="ar-visual-thumb" aria-hidden>
                        <Preview id={item.id} />
                      </span>
                      <span className="ar-visual-body">
                        <span className="ar-visual-top">
                          <strong>{item.name}</strong>
                          <EvidenceBadge evidence={item.evidence} />
                        </span>
                        <em>{item.summary}</em>
                        <span className="ar-visual-foot">
                          <Badge tone="grey">{item.milestone}</Badge>
                          <span className="cv-meta">{item.cohort}</span>
                        </span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          {visibleFiles.length > 0 && (
            <>
              {tab === 'all' && <span className="ar-section">Committed files</span>}
              <ul className="ar-grid">
                {visibleFiles.map((artifact) => (
                  <li key={artifact.id}>
                    <button
                      type="button"
                      className={`ar-card${artifact.id === selectedFile ? ' is-selected' : ''}`}
                      onClick={() => setSelectedFile(artifact.id)}
                      aria-pressed={artifact.id === selectedFile}
                    >
                      <span className="ar-card-head">
                        <span className="ar-card-icon" aria-hidden>
                          {KIND_ICON[artifact.kind]}
                        </span>
                        <span className="ar-card-kind">{artifact.kind}</span>
                        <EvidenceBadge evidence={artifact.evidence} />
                      </span>
                      <strong className="ar-card-name">{artifact.name}</strong>
                      <code className="ar-card-path">{artifact.path}</code>
                      <span className="ar-card-foot">
                        <Badge tone="grey">{artifact.milestone}</Badge>
                        {artifact.sha256 && (
                          <span className="ar-card-sealed">
                            <ShieldCheck size={11} strokeWidth={2.2} aria-hidden /> hashed
                          </span>
                        )}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Card>

        {/* -------------------- inspector ---------------------------- */}
        <Card index={2} hover={false} className="ar-inspector">
          {file ? (
            <FileInspector id={file.id} onBack={() => setSelectedFile(null)} />
          ) : (
            <VisualInspector visual={visual} />
          )}
        </Card>
      </div>
    </PageTransition>
  )
}

/* ---------------------------------------------------------------------- */

function VisualInspector({ visual }: { visual: VisualArtifact }) {
  return (
    <>
      <CardHead
        title={visual.name}
        sub={visual.milestone}
        action={<EvidenceBadge evidence={visual.evidence} />}
      />

      <div className="ar-visual-full">
        <FullVisual id={visual.id} />
      </div>

      <p className="cv-body ar-insp-desc">{visual.summary}</p>

      <div className="ar-insp-block">
        <span className="cv-section-title">What the metric means</span>
        <p className="cv-meta ar-metric">{visual.metric}</p>
      </div>

      <div className="cv-note cv-note--amber ar-caution">
        <Info size={13} strokeWidth={2} aria-hidden />
        <span>{visual.caution}</span>
      </div>

      <div className="ar-insp-block">
        <span className="cv-section-title">Provenance</span>
        <dl className="ar-provenance">
          <Row k="Milestone" v={visual.milestone} />
          <Row k="Cohort" v={visual.cohort} />
          <Row k="Evidence" v={visual.evidence} />
          <Row k="Catalogue" v={`v${freeze.provenance.catalogue_version}`} />
        </dl>
        <div className="ar-hash">
          <code className="cv-hash">{visual.source}</code>
          <CopyButton value={visual.source} label="source path" />
        </div>
      </div>
    </>
  )
}

function FileInspector({ id, onBack }: { id: string; onBack: () => void }) {
  const selected = artifacts.find((a) => a.id === id)!
  return (
    <>
      <CardHead
        icon={KIND_ICON[selected.kind]}
        title={selected.name}
        sub={selected.milestone}
        action={<EvidenceBadge evidence={selected.evidence} />}
      />

      <code className="ar-insp-path">{selected.path}</code>
      <p className="cv-body ar-insp-desc">{selected.description}</p>

      {selected.contents && (
        <div className="ar-insp-block">
          <span className="cv-section-title">Contents</span>
          <ul className="ar-contents">
            {selected.contents.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {selected.sha256 && (
        <div className="ar-insp-block">
          <span className="cv-section-title">SHA-256</span>
          <div className="ar-hash">
            <code className="cv-hash">{selected.sha256}</code>
            <CopyButton value={selected.sha256} label="SHA-256" />
          </div>
        </div>
      )}

      <div className="ar-insp-block">
        <span className="cv-section-title">Provenance</span>
        <dl className="ar-provenance">
          <Row k="Cohort fingerprint" v={`${setc.provenance.cohort_fingerprint.slice(0, 18)}…`} />
          <Row k="Split hash" v={`${freeze.split.train_index_hash.slice(0, 18)}…`} />
          <Row k="Catalogue" v={`v${freeze.provenance.catalogue_version}`} />
          <Row k="Python" v={freeze.provenance.python} />
          <Row
            k="Packages"
            v={`numpy ${freeze.provenance.package_versions.numpy} · xgboost ${freeze.provenance.package_versions.xgboost}`}
          />
        </dl>
      </div>

      <div className="ar-insp-actions">
        <Button variant="ghost" onClick={onBack}>
          Back to visuals
        </Button>
        <Button
          variant="secondary"
          icon={<Download size={14} />}
          href={repoFileUrl(selected.path)}
          external
          title={`Open ${selected.path} in the repository`}
        >
          Open in repository
        </Button>
      </div>
    </>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="cv-kv">
      <dt>{k}</dt>
      <dd className="cv-mono">{v}</dd>
    </div>
  )
}
