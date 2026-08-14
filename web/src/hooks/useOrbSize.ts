import { useEffect, useState } from 'react'

/**
 * Scale an orb down as its column narrows.
 *
 * The steps are *ratios*, not absolute pixel sizes. Each page passes its own
 * canonical size — the Overview hero orb is 412px, the Reliability Report
 * emblem is 168px — and every one of them must shrink proportionally. Returning
 * fixed pixel sizes here would enlarge the smaller orbs on narrow viewports and
 * push them straight out of their containers.
 *
 * Breakpoints mirror the hero grids in the page stylesheets.
 */
const STEPS: readonly { query: string; scale: number }[] = [
  { query: '(max-width: 1180px)', scale: 0.73 },
  { query: '(max-width: 1400px)', scale: 0.8 },
  { query: '(max-width: 1560px)', scale: 0.9 },
]

export function useOrbSize(full = 412): number {
  const [size, setSize] = useState(full)

  useEffect(() => {
    const lists = STEPS.map((step) => window.matchMedia(step.query))

    const resolve = () => {
      const hit = STEPS.findIndex((_, i) => lists[i].matches)
      setSize(hit === -1 ? full : Math.round(full * STEPS[hit].scale))
    }

    resolve()
    lists.forEach((list) => list.addEventListener('change', resolve))
    return () => lists.forEach((list) => list.removeEventListener('change', resolve))
  }, [full])

  return size
}
