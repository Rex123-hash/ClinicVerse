import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Bell, Search } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import CliniverseMark from '../brand/CliniverseMark'
import { product, limitations, setCSpent, provenanceBlemish } from '../../data/cliniverseResults'

/**
 * The header notices are real standing project notices, not invented alerts:
 * the spent holdout, the disclosed provenance blemish, and the selection margin
 * that travels with the frozen pattern.
 */
const NOTICES = [
  { title: 'Set C is spent', body: setCSpent.text },
  { title: 'Disclosed provenance blemish', body: provenanceBlemish.text },
  { title: 'Selection margin', body: limitations[1].text },
]

const SEARCH_TARGETS = [
  { to: '/overview', label: 'Overview', terms: 'confirmed result BUN glucose sodium Na holdout' },
  { to: '/model-lab', label: 'Model Lab', terms: 'frozen model calibrator features XGBoost' },
  { to: '/stress-lab', label: 'Stress Lab', terms: 'withholding replay severity candidates controls' },
  { to: '/reliability-report', label: 'Reliability Report', terms: 'AUROC NLL Brier calibration report' },
  { to: '/experiments', label: 'Experiments', terms: 'milestones history freeze set C' },
  { to: '/artifacts', label: 'Artifacts', terms: 'files evidence case explorer visual tables downloads' },
] as const

export default function TopHeader() {
  const navigate = useNavigate()
  const [noticesOpen, setNoticesOpen] = useState(false)
  const [query, setQuery] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)
  const noticesRef = useRef<HTMLDivElement>(null)
  const searchBoxRef = useRef<HTMLDivElement>(null)
  const searchResults = query.trim()
    ? SEARCH_TARGETS.filter((target) =>
        `${target.label} ${target.terms}`.toLowerCase().includes(query.trim().toLowerCase()),
      )
    : []

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        searchRef.current?.focus()
      }
      if (event.key === 'Escape') setNoticesOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (!noticesOpen) return
    function onClick(event: MouseEvent) {
      if (!noticesRef.current?.contains(event.target as Node)) setNoticesOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [noticesOpen])

  useEffect(() => {
    if (!query) return
    function onClick(event: MouseEvent) {
      if (!searchBoxRef.current?.contains(event.target as Node)) setQuery('')
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [query])

  function openSearchTarget(to: string) {
    setQuery('')
    navigate(to)
  }

  return (
    <header className="cv-header">
      <div className="cv-header-brand">
        <CliniverseMark size={26} />
        <span className="cv-wordmark">{product.name}</span>
        <span className="cv-tagline">{product.tagline}</span>
      </div>

      <div className="cv-search" ref={searchBoxRef}>
        <Search size={15} strokeWidth={1.9} className="cv-search-icon" aria-hidden />
        <input
          ref={searchRef}
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && searchResults[0]) openSearchTarget(searchResults[0].to)
            if (event.key === 'Escape') {
              setQuery('')
              event.currentTarget.blur()
            }
          }}
          placeholder="Search milestones, artifacts, analytes, metrics…"
          aria-label="Search"
          aria-expanded={Boolean(query)}
          aria-controls="cv-search-results"
        />
        <kbd className="cv-kbd" aria-hidden>
          ⌘K
        </kbd>
        {query && (
          <div className="cv-search-results" id="cv-search-results" role="listbox" aria-label="Search results">
            {searchResults.length ? (
              searchResults.map((target) => (
                <button
                  key={target.to}
                  type="button"
                  role="option"
                  aria-selected="false"
                  onClick={() => openSearchTarget(target.to)}
                >
                  <Search size={13} aria-hidden />
                  <span>{target.label}</span>
                  <em>{target.terms}</em>
                </button>
              ))
            ) : (
              <p>No matching route or evidence surface.</p>
            )}
          </div>
        )}
      </div>

      <div className="cv-header-right">
        <div className="cv-notices" ref={noticesRef}>
          <button
            type="button"
            className="cv-icon-btn"
            aria-label={`Project notices (${NOTICES.length})`}
            aria-expanded={noticesOpen}
            onClick={() => setNoticesOpen((open) => !open)}
          >
            <Bell size={17} strokeWidth={1.8} aria-hidden />
            <span className="cv-badge-count">{NOTICES.length}</span>
          </button>

          <AnimatePresence>
            {noticesOpen && (
              <motion.div
                className="cv-notices-panel"
                initial={{ opacity: 0, y: -6, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.98 }}
                transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                role="dialog"
                aria-label="Project notices"
              >
                <div className="cv-notices-head">Standing project notices</div>
                {NOTICES.map((notice) => (
                  <div className="cv-notice" key={notice.title}>
                    <strong>{notice.title}</strong>
                    <p>{notice.body}</p>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="cv-user" aria-label="Repository owner: Amaan Khan">
          <span className="cv-avatar" aria-hidden>
            AK
          </span>
          <span className="cv-user-text">
            <strong>Amaan Khan</strong>
            <em>Repository owner</em>
          </span>
        </div>
      </div>
    </header>
  )
}
