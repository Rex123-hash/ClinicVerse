import { useEffect, useRef, useState } from 'react'
import { useInView } from 'framer-motion'

/**
 * Counts a value up once, the first time it scrolls into view.
 *
 * The displayed value starts *at* the target, so the real number is on screen
 * even if the tween never runs — an off-screen mount, a paused tab, or reduced
 * motion must never leave a scientific figure reading zero.
 */
export function useCountUp(target: number, places: number, duration = 900) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: '-40px' })
  const [value, setValue] = useState(target)
  const done = useRef(false)

  useEffect(() => {
    if (!inView || done.current) return
    done.current = true

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target)
      return
    }

    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(target * eased)
      if (t < 1) frame = requestAnimationFrame(tick)
      else setValue(target)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      setValue(target)
    }
  }, [inView, target, duration])

  return { ref, text: value.toFixed(places) }
}
