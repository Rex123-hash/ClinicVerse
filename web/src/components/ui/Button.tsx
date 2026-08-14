import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'dark'

interface Common {
  variant?: Variant
  icon?: ReactNode
  trailing?: ReactNode
  /** Why the control is unavailable. Rendered as the disabled explanation. */
  disabledReason?: string
  children: ReactNode
  className?: string
}

type AsButton = Common &
  Omit<ButtonHTMLAttributes<HTMLButtonElement>, keyof Common> & { href?: undefined }

type AsLink = Common &
  Omit<AnchorHTMLAttributes<HTMLAnchorElement>, keyof Common> & {
    /** Renders an anchor instead of a button. */
    href: string
    /** Open in a new tab, always with `noopener noreferrer`. */
    external?: boolean
  }

type Props = AsButton | AsLink

/** Props this component consumes itself and must not forward to the DOM. */
const OWN = new Set(['variant', 'icon', 'trailing', 'disabledReason', 'children', 'className', 'external'])

function passthrough(props: Props): Record<string, unknown> {
  return Object.fromEntries(Object.entries(props).filter(([key]) => !OWN.has(key)))
}

export default function Button(props: Props) {
  const { variant = 'secondary', icon, trailing, disabledReason, children, className = '' } = props
  const classes = `cv-btn cv-btn--${variant} ${className}`

  const inner = (
    <>
      {icon && <span className="cv-btn-icon">{icon}</span>}
      <span>{children}</span>
      {trailing && <span className="cv-btn-trailing">{trailing}</span>}
    </>
  )

  // An anchor carries the action natively: real middle-click, real "copy link
  // address", keyboard activation for free, and a `download` the browser
  // honours without any script running.
  if (props.href !== undefined) {
    return (
      <a
        {...passthrough(props)}
        className={classes}
        {...(props.external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      >
        {inner}
      </a>
    )
  }

  const isDisabled = props.disabled ?? Boolean(disabledReason)
  return (
    <button
      {...passthrough(props)}
      type="button"
      className={classes}
      disabled={isDisabled}
      title={isDisabled ? disabledReason : props.title}
    >
      {inner}
    </button>
  )
}
