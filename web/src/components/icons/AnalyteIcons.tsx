/**
 * The analyte icon family.
 *
 * One visual language across every glyph: a 24×24 box, 1.6px strokes,
 * round caps and joins, drawn in `currentColor` so the chip controls the
 * colour and the hover state. No letters standing in for icons, no raster
 * assets, and deliberately no depiction of a biological mechanism — these
 * mark a laboratory measurement, not a physiological claim.
 */

interface IconProps {
  size?: number
  className?: string
}

const BASE = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

/** Blood urea nitrogen — a specimen tube with a measured fill line. */
export function AnalyteTubeIcon({ size = 21, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} aria-hidden {...BASE}>
      <path d="M9 3h6" />
      <path d="M10 3v13a2 2 0 0 0 4 0V3" />
      <path d="M10 12.5h4" />
      <path d="M17.5 6.5h3" />
      <path d="M17.5 10h2" />
    </svg>
  )
}

/** Serum glucose — a restrained specimen droplet with a level marker. */
export function AnalyteDropletIcon({ size = 21, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} aria-hidden {...BASE}>
      <path d="M12 3.5c3 3.6 5 6.3 5 8.8a5 5 0 0 1-10 0c0-2.5 2-5.2 5-8.8Z" />
      <path d="M8.4 13.6h7.2" />
    </svg>
  )
}

/** Sodium — a charged particle: nucleus, orbital shell and a charge mark. */
export function AnalyteIonIcon({ size = 21, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} aria-hidden {...BASE}>
      <circle cx="11" cy="12.5" r="2.2" />
      <ellipse cx="11" cy="12.5" rx="7.5" ry="4" transform="rotate(-28 11 12.5)" />
      <path d="M17.8 5.4v3.2M16.2 7h3.2" />
    </svg>
  )
}

/**
 * The reliability core mark: a thin hexagonal container around a centred
 * clinical cross. Drawn on a 44×48 grid so the hexagon is mathematically
 * regular and the cross sits on its exact centre.
 */
export function ReliabilityCoreIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 44 48" fill="none" className={className} aria-hidden>
      <path
        d="M22 2.6 L39.2 12.55 V32.45 L22 42.4 L4.8 32.45 V12.55 Z"
        fill="rgba(0,32,40,0.5)"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="M22 15.2 V29.8 M14.7 22.5 H29.3"
        stroke="currentColor"
        strokeWidth="2.1"
        strokeLinecap="round"
      />
    </svg>
  )
}
