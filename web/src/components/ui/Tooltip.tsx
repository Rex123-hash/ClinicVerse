import { useId, useState, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Info } from 'lucide-react'
import { glossary } from '../../data/cliniverseResults'

/**
 * An inline definition for an unfamiliar scientific label. Definitions live in
 * the glossary of the shared data module so wording stays consistent.
 */
export default function Tooltip({
  term,
  text,
  children,
}: {
  term?: string
  text?: string
  children?: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const body = text ?? (term ? glossary[term] : undefined)
  if (!body) return <>{children}</>

  return (
    <span className="cv-tip-wrap">
      {children}
      <button
        type="button"
        className="cv-tip-btn"
        aria-label={`What is ${term ?? 'this'}?`}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        <Info size={12} strokeWidth={2} aria-hidden />
      </button>
      <AnimatePresence>
        {open && (
          <motion.span
            id={id}
            role="tooltip"
            className="cv-tip"
            initial={{ opacity: 0, y: 4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.97 }}
            transition={{ duration: 0.15 }}
          >
            {term && <strong>{term}</strong>}
            {body}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  )
}
