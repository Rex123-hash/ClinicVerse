import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Check } from 'lucide-react'
import type { TimelineStep } from '../../data/cliniverseResults'

/** Horizontal investigation timeline; the connector reveals left-to-right. */
export default function Timeline({ steps }: { steps: readonly TimelineStep[] }) {
  const ref = useRef<HTMLOListElement>(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })

  return (
    <ol className="cv-timeline" ref={ref}>
      <motion.span
        className="cv-timeline-rail"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: inView ? 1 : 0 }}
        transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
        aria-hidden
      />
      {steps.map((step, i) => (
        <motion.li
          key={step.label}
          className={`cv-tl-step${step.state === 'active' ? ' is-active' : ''}`}
          initial={{ opacity: 0, y: 6 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.35, delay: 0.12 + i * 0.09, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="cv-tl-node" aria-hidden>
            {step.state === 'active' ? (
              <i className="cv-tl-pulse" />
            ) : (
              <Check size={11} strokeWidth={3} />
            )}
          </span>
          <span className="cv-tl-body">
            <strong>{step.label}</strong>
            <em>{step.detail}</em>
            <span className="cv-tl-date">{step.date}</span>
          </span>
        </motion.li>
      ))}
    </ol>
  )
}
