import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Search } from 'lucide-react'
import { visual, type PatientRow } from '../../data/cliniverseResults'

/**
 * Case Explorer.
 *
 * The committed per-patient rows from `setc_oneshot_predictions.npz`: source
 * record id, recorded outcome, and the saved clean/withheld predictions. The
 * source id is replaced by a deterministic public alias at this UI boundary.
 *
 * There are deliberately no clinical covariates here. The artifact stores
 * predictions and outcomes only — no ages, no vitals, no diagnoses — and this
 * table shows exactly what the artifact holds and nothing more.
 */

type SortKey = 'id' | 'c' | 'w' | 'd' | 'rc'

const PAGE = 12

const CASE_ALIASES = new Map(
  [...visual.patients]
    .sort((a, b) => a.id - b.id)
    .map((patient, index) => [patient.id, `CV-${String(index + 1).padStart(4, '0')}`]),
)

function caseAlias(patient: PatientRow): string {
  return CASE_ALIASES.get(patient.id) ?? 'CV-UNLISTED'
}

export default function CaseExplorer({ compact = false }: { compact?: boolean }) {
  const [sort, setSort] = useState<SortKey>('d')
  const [desc, setDesc] = useState(true)
  const [page, setPage] = useState(0)
  const [query, setQuery] = useState('')
  const [outcome, setOutcome] = useState<'all' | 'died' | 'survived'>('all')

  const rows = useMemo(() => {
    let list: PatientRow[] = visual.patients
    if (outcome !== 'all') list = list.filter((p) => (outcome === 'died' ? p.y === 1 : p.y === 0))
    if (query.trim()) {
      const needle = query.trim().toUpperCase()
      list = list.filter((p) => caseAlias(p).includes(needle))
    }
    return [...list].sort((a, b) => (desc ? b[sort] - a[sort] : a[sort] - b[sort]))
  }, [sort, desc, query, outcome])

  const pages = Math.max(1, Math.ceil(rows.length / PAGE))
  const current = Math.min(page, pages - 1)
  const slice = rows.slice(current * PAGE, current * PAGE + PAGE)

  function toggle(key: SortKey) {
    if (key === sort) setDesc((d) => !d)
    else {
      setSort(key)
      setDesc(true)
    }
    setPage(0)
  }

  const Th = ({ k, label }: { k: SortKey; label: string }) => (
    <th>
      <button type="button" className="vz-sort" onClick={() => toggle(k)}>
        {label}
        {sort === k &&
          (desc ? <ArrowDown size={11} strokeWidth={2.4} /> : <ArrowUp size={11} strokeWidth={2.4} />)}
      </button>
    </th>
  )

  return (
    <div className="vz-cases">
      {!compact && (
        <div className="vz-cases-bar">
          <label className="vz-cases-search">
            <Search size={13} strokeWidth={2} aria-hidden />
            <input
              type="search"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setPage(0)
              }}
              placeholder="Case alias…"
              aria-label="Filter by case alias"
            />
          </label>
          <div className="vz-cases-filters" role="tablist">
            {(['all', 'died', 'survived'] as const).map((o) => (
              <button
                key={o}
                type="button"
                role="tab"
                aria-selected={outcome === o}
                className={`vz-cases-filter${outcome === o ? ' is-active' : ''}`}
                onClick={() => {
                  setOutcome(o)
                  setPage(0)
                }}
              >
                {o === 'all' ? 'All' : o === 'died' ? 'Died' : 'Survived'}
              </button>
            ))}
          </div>
          <span className="cv-meta">{rows.length.toLocaleString('en-US')} rows</span>
        </div>
      )}

      <div className="cv-table-scroll">
        <table className="cv-table vz-cases-table">
          <thead>
            <tr>
              <Th k="id" label="Case" />
              <th>Outcome</th>
              <Th k="c" label="Clean risk" />
              <Th k="w" label="Withheld risk" />
              <Th k="d" label="Excess NLL" />
              <Th k="rc" label="Cells" />
            </tr>
          </thead>
          <tbody>
            {slice.map((p) => {
              const shift = p.w - p.c
              return (
                <tr key={p.id}>
                  <td className="cv-mono">{caseAlias(p)}</td>
                  <td>
                    <span className={`vz-outcome${p.y === 1 ? ' is-death' : ''}`}>
                      {p.y === 1 ? 'Died' : 'Survived'}
                    </span>
                  </td>
                  <td>{p.c.toFixed(5)}</td>
                  <td>
                    {p.w.toFixed(5)}
                    <em className={`vz-shift${shift < 0 ? ' is-down' : ''}`}>
                      {shift > 0 ? '+' : '−'}
                      {Math.abs(shift).toFixed(4)}
                    </em>
                  </td>
                  <td className={p.d > 0 ? 'vz-pos' : ''}>
                    {p.d > 0 ? '+' : '−'}
                    {Math.abs(p.d).toFixed(5)}
                  </td>
                  <td>{p.rc}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {!compact && (
        <div className="vz-cases-pager">
          <button type="button" onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={current === 0}>
            Previous
          </button>
          <span className="cv-meta">
            page {current + 1} of {pages.toLocaleString('en-US')}
          </span>
          <button
            type="button"
            onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
            disabled={current >= pages - 1}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
