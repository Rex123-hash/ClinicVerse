import { useState } from 'react'
import { Check, Copy } from 'lucide-react'

/** Copies a hash or path, with visible confirmation. */
export default function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard unavailable — the value stays selectable in the DOM */
    }
  }

  return (
    <button
      type="button"
      className={`cv-copy${copied ? ' is-copied' : ''}`}
      onClick={copy}
      aria-label={copied ? `${label} copied` : `Copy ${label}`}
    >
      {copied ? <Check size={12} strokeWidth={2.4} /> : <Copy size={12} strokeWidth={2} />}
      <span>{copied ? 'Copied' : 'Copy'}</span>
    </button>
  )
}
