/** Shown while a route chunk loads. Mirrors the page rhythm so nothing jumps. */
export default function PageSkeleton() {
  return (
    <div className="cv-page" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading…</span>
      <div className="cv-skel cv-skel-title" />
      <div className="cv-skel cv-skel-hero" />
      <div className="cv-skel-row">
        <div className="cv-skel cv-skel-card" />
        <div className="cv-skel cv-skel-card" />
        <div className="cv-skel cv-skel-card" />
      </div>
    </div>
  )
}
