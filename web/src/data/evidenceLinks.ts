/**
 * Where the application points when a judge asks "show me the actual file".
 *
 * Two mechanisms, and the distinction matters:
 *
 *   - Committed markdown reports and large binary artifacts are *linked* to the
 *     public repository. They are read where they live, so nothing here can
 *     drift from the reviewed source.
 *   - The set-c confirmation result is *served* from `public/evidence/`, copied
 *     byte-for-byte by `scripts/export_ui_data.py`, which refuses to run if the
 *     artifact's digest stops matching the one the Artifacts page displays.
 *     Downloading it and hashing it must reproduce that digest exactly.
 *
 * This file adds no scientific values. Every path below already exists in the
 * repository and is already named somewhere in the UI.
 */

const REPOSITORY = 'https://github.com/Rex123-hash/ClinicVerse'

/** Committed files are linked on the default branch, which is frozen. */
const REF = 'main'

/**
 * Public URL for a repository-relative path, e.g. `docs/M5_V2_DESIGN.md`.
 *
 * Two artifact entries name a directory of committed figures rather than a
 * single file. GitHub serves those under `/tree/`, not `/blob/`, so a trailing
 * slash selects the right form instead of producing a 404.
 */
export function repoFileUrl(path: string): string {
  const isDirectory = path.endsWith('/')
  const kind = isDirectory ? 'tree' : 'blob'
  return `${REPOSITORY}/${kind}/${REF}/${isDirectory ? path.slice(0, -1) : path}`
}

/**
 * The one-shot set-c confirmation result, served verbatim.
 *
 * `sha256` is the digest of the committed artifact and of this download; they
 * are the same bytes. The exporter asserts it on every run.
 */
export const setCEvidenceFile = {
  /** Same-origin, so the `download` attribute is honoured without script. */
  url: '/evidence/setc-confirmation.json',
  /** What the browser saves it as. */
  filename: 'cliniverse-set-c-confirmation.json',
  /** Where the identical bytes live in the repository. */
  sourcePath: 'experiments/robustness/results/m5v2_setc/results.json',
  sha256: '7179a5744e5d9034a735fb6bcd1652a96e850e285fc60b8de61983a7d192a907',
  bytes: 6739,
} as const

/** The written report behind the confirmed result. */
export const setCReportPath = 'docs/M5_V2_SETC_CONFIRMATION.md'
