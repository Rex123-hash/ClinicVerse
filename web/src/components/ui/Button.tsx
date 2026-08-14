import type { ButtonHTMLAttributes, ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'dark'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  icon?: ReactNode
  trailing?: ReactNode
  /** Why the control is unavailable. Rendered as the disabled explanation. */
  disabledReason?: string
  children: ReactNode
}

export default function Button({
  variant = 'secondary',
  icon,
  trailing,
  disabledReason,
  disabled,
  children,
  className = '',
  ...rest
}: Props) {
  const isDisabled = disabled ?? Boolean(disabledReason)
  return (
    <button
      type="button"
      className={`cv-btn cv-btn--${variant} ${className}`}
      disabled={isDisabled}
      title={isDisabled ? disabledReason : rest.title}
      aria-describedby={undefined}
      {...rest}
    >
      {icon && <span className="cv-btn-icon">{icon}</span>}
      <span>{children}</span>
      {trailing && <span className="cv-btn-trailing">{trailing}</span>}
    </button>
  )
}
