import type { ReactNode } from 'react'
import type { EvidenceClass } from '../../data/cliniverseResults'
import { evidenceLabel } from '../../data/cliniverseResults'

export type BadgeTone =
  | 'teal'
  | 'green'
  | 'orange'
  | 'amber'
  | 'navy'
  | 'grey'
  | 'mint'

export function Badge({
  tone = 'grey',
  children,
  icon,
  title,
}: {
  tone?: BadgeTone
  children: ReactNode
  icon?: ReactNode
  title?: string
}) {
  return (
    <span className={`cv-badge cv-badge--${tone}`} title={title}>
      {icon}
      {children}
    </span>
  )
}

const EVIDENCE_TONE: Record<EvidenceClass, BadgeTone> = {
  confirmed: 'green',
  development: 'teal',
  descriptive: 'grey',
  historical: 'navy',
  limitation: 'amber',
}

/** Renders how strongly a claim is supported — never decorative. */
export function EvidenceBadge({ evidence }: { evidence: EvidenceClass }) {
  return <Badge tone={EVIDENCE_TONE[evidence]}>{evidenceLabel[evidence]}</Badge>
}
