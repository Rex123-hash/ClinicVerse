import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'

type CardProps = Omit<HTMLMotionProps<'section'>, 'title'> & {
  /** Card entry order within a page-level stagger. */
  index?: number
  /** Lift on hover. Off for dense panels that carry their own interactions. */
  hover?: boolean
  tone?: 'default' | 'dark' | 'soft'
  children: ReactNode
}

export function Card({
  index = 0,
  hover = true,
  tone = 'default',
  className = '',
  children,
  ...rest
}: CardProps) {
  return (
    <motion.section
      className={`cv-card cv-card--${tone}${hover ? ' cv-card--hover' : ''} ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.42,
        delay: Math.min(index * 0.045, 0.36),
        ease: [0.16, 1, 0.3, 1],
      }}
      {...rest}
    >
      {children}
    </motion.section>
  )
}

export function CardHead({
  title,
  sub,
  icon,
  action,
  live,
  status,
}: {
  title: ReactNode
  sub?: ReactNode
  icon?: ReactNode
  action?: ReactNode
  live?: boolean
  status?: string
}) {
  return (
    <header className="cv-card-head">
      {icon && <span className="cv-card-icon">{icon}</span>}
      <span className="cv-card-headtext">
        <span className="cv-card-title">
          {title}
          {(live || status) && <LiveBadge label={status ?? 'Live'} />}
        </span>
        {sub && <span className="cv-meta">{sub}</span>}
      </span>
      {action && <span className="cv-card-action">{action}</span>}
    </header>
  )
}

export function LiveBadge({ label = 'Live' }: { label?: string }) {
  return (
    <span className="cv-live">
      <i className="cv-live-dot" aria-hidden />
      {label}
    </span>
  )
}
