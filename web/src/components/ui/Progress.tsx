import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

/** Fills from zero to its real value when it first enters the viewport. */
export default function Progress({
  value,
  tone = 'teal',
  height = 5,
  label,
}: {
  /** 0 – 1 */
  value: number
  tone?: 'teal' | 'green' | 'orange' | 'navy'
  height?: number
  label?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-30px' })
  const pct = Math.max(0, Math.min(1, value))

  return (
    <div
      ref={ref}
      className={`cv-progress cv-progress--${tone}`}
      style={{ height }}
      role="progressbar"
      aria-valuenow={Math.round(pct * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <motion.span
        className="cv-progress-fill"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: inView ? pct : 0 }}
        transition={{ duration: 0.85, ease: [0.16, 1, 0.3, 1] }}
      />
    </div>
  )
}
