/** The Cliniverse mark: a teal ring opening to the right around a clinical cross. */
export default function CliniverseMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label="Cliniverse"
    >
      <defs>
        <linearGradient id="cv-mark-g" x1="4" y1="3" x2="27" y2="29" gradientUnits="userSpaceOnUse">
          <stop stopColor="#32C7C0" />
          <stop offset="0.55" stopColor="#10AAA5" />
          <stop offset="1" stopColor="#087D7D" />
        </linearGradient>
      </defs>
      <path
        d="M25.6 8.4A11.4 11.4 0 1 0 25.6 23.6"
        stroke="url(#cv-mark-g)"
        strokeWidth="4.3"
        strokeLinecap="round"
      />
      <path
        d="M24.4 12.1v7.8M20.5 16h7.8"
        stroke="url(#cv-mark-g)"
        strokeWidth="3.1"
        strokeLinecap="round"
      />
    </svg>
  )
}
