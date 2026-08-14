import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react'
import { animate, motion, useMotionValue, useReducedMotion, useSpring, useTransform } from 'framer-motion'
import {
  ACCENT_ARCS,
  BOX,
  CENTER,
  ECG_PATH,
  MICRO_NODES,
  ORBITS,
  RINGS,
  TICKS,
  arcPath,
  buildOrbitNodes,
  buildParticles,
  ellipsePath,
  type Orbit,
  type OrbitNode,
  type Ring,
} from './orbGeometry'
import { ReliabilityCoreIcon } from '../icons/AnalyteIcons'
import './orb.css'

export type OrbVariant =
  | 'overview'
  | 'model'
  | 'stress'
  | 'report'
  | 'experiment'
  | 'artifact'

export interface OrbNode {
  id: string
  label: string
  sub?: string
  /** A 24px SVG from the analyte icon family. Never a letter or emoji. */
  icon?: ReactNode
  /** Position on the orb box, 0–1 in each axis. */
  x: number
  y: number
  /** Chip width in px. The reference chips are deliberately unequal. */
  width?: number
}

interface Props {
  variant?: OrbVariant
  /** Total visual footprint including the outer orbital rings. */
  size?: number
  nodes?: readonly OrbNode[]
  caption?: ReactNode
  /** 0–1. Drives ring speed, arc frequency and glow strength. */
  intensity?: number
  /** Optional visual-only wave duration. */
  ecgDuration?: number
  className?: string
  label: string
}

const VARIANTS: Record<
  OrbVariant,
  {
    speed: number
    breath: number
    ecg: number
    arcs: boolean
    coreRatio: number
    /** How strongly this orb answers press and click. */
    response: number
  }
> = {
  overview: { speed: 1, breath: 3.8, ecg: 7, arcs: false, coreRatio: 0.49, response: 1 },
  model: { speed: 0.72, breath: 4.6, ecg: 4.4, arcs: false, coreRatio: 0.5, response: 0.55 },
  stress: { speed: 1.045, breath: 3.8, ecg: 7, arcs: true, coreRatio: 0.48, response: 1.45 },
  report: { speed: 0.6, breath: 5.2, ecg: 8.5, arcs: false, coreRatio: 0.52, response: 0.45 },
  experiment: { speed: 1.25, breath: 3.6, ecg: 5.5, arcs: false, coreRatio: 0.49, response: 0.85 },
  artifact: { speed: 0.75, breath: 4.4, ecg: 7.5, arcs: false, coreRatio: 0.5, response: 0.55 },
}

/**
 * Differential spring character per depth layer *inside the core*. The specular
 * highlight leads, the glass body follows, and the inner rim trails — that lag
 * is what reads as a translucent sphere rather than a flat disc.
 */
const SPRING = {
  specular: { stiffness: 300, damping: 20, mass: 0.6 },
  particles: { stiffness: 220, damping: 24, mass: 0.8 },
  symbol: { stiffness: 190, damping: 26, mass: 0.9 },
  body: { stiffness: 160, damping: 24, mass: 1 },
  rim: { stiffness: 110, damping: 26, mass: 1.15 },
  /** Under-damped: gives 2–3 small settling oscillations, not a cartoon bounce. */
  press: { stiffness: 260, damping: 16, mass: 0.85 },
} as const

const ORBIT_NODES = buildOrbitNodes()
const PARTICLES = buildParticles()

/** Transparent stroke width that makes a 1px path comfortably hoverable. */
const HIT_STROKE = 14

interface Ripple {
  key: number
  cx: number
  cy: number
}

export default function CliniverseOrb({
  variant = 'overview',
  size = 410,
  nodes = [],
  caption,
  intensity = 0.5,
  ecgDuration,
  className = '',
  label,
}: Props) {
  const config = VARIANTS[variant]
  const reduced = useReducedMotion() ?? false
  const coreRef = useRef<HTMLDivElement>(null)
  const ecgTravelRef = useRef<SVGGElement>(null)
  const ecgRateRef = useRef(1)
  const [ripples, setRipples] = useState<Ripple[]>([])
  const rippleId = useRef(0)

  /* ------------------------------------------------------------------ */
  /* independent interaction state — one per hit target, never global    */
  /* ------------------------------------------------------------------ */

  const [hoveredAnalyte, setHoveredAnalyte] = useState<string | null>(null)
  const [hoveredOrbit, setHoveredOrbit] = useState<string | null>(null)
  const [coreActive, setCoreActive] = useState(false)
  const [visible, setVisible] = useState(true)

  const uid = variant
  const core = Math.round(size * config.coreRatio)
  const speed = variant === 'stress' ? config.speed : config.speed * (0.62 + intensity * 0.85)
  const play = visible ? 'running' : 'paused'
  const response = config.response

  /* ---- core-local springs: driven ONLY by the core hit target ---- */

  const bodyX = useSpring(0, SPRING.body)
  const bodyY = useSpring(0, SPRING.body)
  const specX = useSpring(0, SPRING.specular)
  const specY = useSpring(0, SPRING.specular)
  const falloffX = useSpring(0, SPRING.body)
  const falloffY = useSpring(0, SPRING.body)
  const symbolX = useSpring(0, SPRING.symbol)
  const symbolY = useSpring(0, SPRING.symbol)
  const partX = useSpring(0, SPRING.particles)
  const partY = useSpring(0, SPRING.particles)
  const rimX = useSpring(0, SPRING.rim)
  const rimY = useSpring(0, SPRING.rim)
  const tiltX = useSpring(0, SPRING.body)
  const tiltY = useSpring(0, SPRING.body)
  const pressX = useSpring(1, SPRING.press)
  const pressY = useSpring(1, SPRING.press)
  const ecgPulse = useMotionValue(1)
  const energy = useMotionValue(0)
  const particleGlow = useTransform(energy, (v) => `brightness(${1 + v * 0.85})`)

  /**
   * Aim the core's internal layers at a pointer position expressed in the
   * core's own coordinates (−0.5 … 0.5). Nothing outside the sphere reads
   * these values — the ring system has its own, separate interaction state.
   */
  const aimCore = useCallback(
    (nx: number, ny: number) => {
      bodyX.set(nx * 11 * response)
      bodyY.set(ny * 11 * response)
      specX.set(nx * 30)
      specY.set(ny * 30)
      // the dark falloff travels against the pointer — this sells the depth
      falloffX.set(nx * -18)
      falloffY.set(ny * -18)
      symbolX.set(nx * 4)
      symbolY.set(ny * 4)
      partX.set(nx * 2.4)
      partY.set(ny * 2.4)
      rimX.set(nx * 6)
      rimY.set(ny * 6)
      tiltX.set(-ny * 5)
      tiltY.set(nx * 5)
    },
    [
      bodyX, bodyY, specX, specY, falloffX, falloffY, symbolX, symbolY,
      partX, partY, rimX, rimY, tiltX, tiltY, response,
    ],
  )

  const onCoreMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (reduced) return
      const node = coreRef.current
      if (!node) return
      const rect = node.getBoundingClientRect()
      const nx = (event.clientX - rect.left) / rect.width - 0.5
      const ny = (event.clientY - rect.top) / rect.height - 0.5
      aimCore(nx, ny)
      // the inner light follows the pointer; the outer halo does not move
      node.style.setProperty('--glow-x', `${(50 + nx * 26).toFixed(1)}%`)
      node.style.setProperty('--glow-y', `${(46 + ny * 26).toFixed(1)}%`)
    },
    [aimCore, reduced],
  )

  const onCoreLeave = useCallback(() => {
    setCoreActive(false)
    aimCore(0, 0)
    pressX.set(1)
    pressY.set(1)
    const node = coreRef.current
    if (!node) return
    node.style.setProperty('--glow-x', '50%')
    node.style.setProperty('--glow-y', '46%')
  }, [aimCore, pressX, pressY])

  const onCoreDown = useCallback(() => {
    if (reduced) return
    pressX.set(1 + 0.025 * response)
    pressY.set(1 - 0.025 * response)
  }, [pressX, pressY, reduced, response])

  const fireCoreImpulse = useCallback(
    (cx: number, cy: number) => {
      if (reduced) return
      pressX.set(1)
      pressY.set(1)

      rippleId.current += 1
      const key = rippleId.current
      setRipples((current) => [...current.slice(-3), { key, cx, cy }])
      window.setTimeout(
        () => setRipples((current) => current.filter((r) => r.key !== key)),
        760,
      )

      animate(ecgPulse, [1, 1 + 0.5 * response, 1], {
        duration: 0.72,
        ease: [0.16, 1, 0.3, 1],
      })
      animate(energy, [0, 1, 0], { duration: 0.9, ease: 'easeOut' })
    },
    [reduced, pressX, pressY, ecgPulse, energy, response],
  )

  const onCoreUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const rect = coreRef.current?.getBoundingClientRect()
      if (!rect) return
      fireCoreImpulse(
        ((event.clientX - rect.left) / rect.width) * BOX,
        ((event.clientY - rect.top) / rect.height) * BOX,
      )
    },
    [fireCoreImpulse],
  )

  const onCoreKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if ((event.key !== 'Enter' && event.key !== ' ') || event.repeat) return
      event.preventDefault()
      fireCoreImpulse(CENTER, CENTER)
    },
    [fireCoreImpulse],
  )

  /* ---- pause when off-screen or the tab is hidden ---- */

  useEffect(() => {
    const node = coreRef.current
    if (!node) return
    let onScreen = true
    const sync = () => setVisible(onScreen && !document.hidden)
    const observer = new IntersectionObserver(
      ([entry]) => {
        onScreen = entry.isIntersecting
        sync()
      },
      { rootMargin: '100px' },
    )
    observer.observe(node)
    const onVisibility = () => sync()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      observer.disconnect()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  /* Preserve the continuous wave timeline while severity changes its speed. */
  useEffect(() => {
    if (variant !== 'stress' || ecgDuration == null || reduced) return
    const animation = ecgTravelRef.current?.getAnimations()[0]
    if (!animation) return

    const from = ecgRateRef.current
    const target = config.ecg / ecgDuration
    const startedAt = performance.now()
    const transitionMs = 240
    let frame = 0

    const updateRate = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / transitionMs)
      const eased = 1 - Math.pow(1 - progress, 3)
      const rate = from + (target - from) * eased
      animation.updatePlaybackRate(rate)
      ecgRateRef.current = rate
      if (progress < 1) frame = requestAnimationFrame(updateRate)
    }

    frame = requestAnimationFrame(updateRate)
    return () => cancelAnimationFrame(frame)
  }, [config.ecg, ecgDuration, reduced, variant])

  const rearOrbits = useMemo(() => ORBITS.filter((o) => o.plane === 'rear'), [])
  const frontOrbits = useMemo(() => ORBITS.filter((o) => o.plane === 'front'), [])

  const style = {
    width: size,
    height: size,
    ['--core' as string]: `${core}px`,
    ['--breath' as string]: `${config.breath}s`,
    ['--ecg-dur' as string]: `${(
      variant === 'stress' ? config.ecg : config.ecg / (0.6 + intensity * 0.8)
    ).toFixed(2)}s`,
    ['--glow-boost' as string]: (0.7 + intensity * 0.6).toFixed(2),
  } as CSSProperties

  const orbitHover = {
    hovered: hoveredOrbit,
    onEnter: setHoveredOrbit,
    onLeave: () => setHoveredOrbit(null),
    reduced,
  }

  return (
    <div className={`cv-orb cv-orb--${variant} ${className}`} style={style} role="img" aria-label={label}>
      <div className="orb-stage">
        <svg className="orb-svg" viewBox={`0 0 ${BOX} ${BOX}`} aria-hidden>
          <defs>
            <radialGradient id={`orbCore-${uid}`} cx="42%" cy="31%">
              <stop offset="0%" stopColor="#0A6C70" stopOpacity="0.64" />
              <stop offset="15%" stopColor="#07575F" stopOpacity="0.76" />
              <stop offset="34%" stopColor="#034A54" stopOpacity="0.9" />
              <stop offset="57%" stopColor="#02343F" />
              <stop offset="77%" stopColor="#012A35" />
              <stop offset="91%" stopColor="#001F2B" />
              <stop offset="100%" stopColor="#001722" />
            </radialGradient>

            <linearGradient id="orbRimGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#BFFFF6" stopOpacity="1" />
              <stop offset="34%" stopColor="#4FE8DA" stopOpacity="0.5" />
              <stop offset="68%" stopColor="#0B8C8C" stopOpacity="0.16" />
              <stop offset="100%" stopColor="#8FF3EA" stopOpacity="0.8" />
            </linearGradient>

            <linearGradient id="orbRingGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#A6FFF5" stopOpacity="0.95" />
              <stop offset="45%" stopColor="#32C7C0" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#06616B" stopOpacity="0.1" />
            </linearGradient>

            <radialGradient id="orbHaloGrad">
              <stop offset="0%" stopColor="#5FF0E2" stopOpacity="0.7" />
              <stop offset="100%" stopColor="#0FA9A4" stopOpacity="0.15" />
            </radialGradient>

            <radialGradient id={`orbNode-${uid}`}>
              <stop offset="0%" stopColor="#EAFFFC" />
              <stop offset="45%" stopColor="#7DF2E6" />
              <stop offset="100%" stopColor="#1DA9A4" stopOpacity="0.35" />
            </radialGradient>

            <filter id={`orbSoft-${uid}`} x="-70%" y="-70%" width="240%" height="240%">
              <feGaussianBlur stdDeviation="1.9" />
            </filter>
            <filter id={`orbBloom-${uid}`} x="-90%" y="-90%" width="280%" height="280%">
              <feGaussianBlur stdDeviation="3.4" />
            </filter>
            <filter id={`orbRearSoft-${uid}`} x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="0.55" />
            </filter>
          </defs>

          {/* ============ 1. atmospheric particles (rearmost) ========== */}
          <motion.g filter={`url(#orbSoft-${uid})`} style={{ filter: particleGlow }}>
            {PARTICLES.filter((p) => p.zone === 'atmos').map((p) => (
              <circle
                key={p.id}
                className={`${p.sparkle ? 'orb-sparkle' : 'orb-particle'} cv-orb-motion`}
                cx={p.cx}
                cy={p.cy}
                r={p.r}
                fill={p.sparkle ? '#DFFFFA' : '#7DE8DE'}
                opacity={p.opacity}
                style={{
                  ['--dur' as string]: `${p.duration.toFixed(1)}s`,
                  ['--dx' as string]: `${p.dx.toFixed(1)}px`,
                  ['--dy' as string]: `${p.dy.toFixed(1)}px`,
                  animationDelay: `${p.delay.toFixed(1)}s`,
                  animationPlayState: play,
                }}
              />
            ))}
          </motion.g>

          {/* ============ 2. rear elliptical orbits ==================== */}
          {rearOrbits.map((orbit) => (
            <OrbitLayer key={orbit.id} orbit={orbit} speed={speed} play={play} {...orbitHover} />
          ))}
          <OrbitTravellers plane="rear" uid={uid} speed={speed} play={play} hovered={hoveredOrbit} />

          {/* ============ 3. technical bands ========================== */}
          {RINGS.map((ring) => (
            <RingLayer key={ring.id} ring={ring} uid={uid} speed={speed} play={play} {...orbitHover} />
          ))}

          {ACCENT_ARCS.map((accent) => (
            <Spin key={accent.id} dur={Math.abs(accent.spin) / speed} rev={accent.spin < 0} play={play}>
              <path
                d={arcPath(accent.r, accent.arc[0], accent.arc[1])}
                fill="none"
                stroke="#C8FFF7"
                strokeWidth={accent.width}
                strokeOpacity={accent.opacity}
                strokeLinecap="round"
              />
            </Spin>
          ))}

          <Spin dur={96 / speed} rev={false} play={play}>
            {TICKS.filter((tick) => tick.band !== 2).map((tick) => (
              <line
                key={tick.id}
                x1={tick.x1}
                y1={tick.y1}
                x2={tick.x2}
                y2={tick.y2}
                stroke="#8FF3EA"
                strokeWidth={tick.width}
                strokeOpacity={tick.opacity}
                strokeLinecap="round"
              />
            ))}
          </Spin>
          <Spin dur={128 / speed} rev play={play}>
            {TICKS.filter((tick) => tick.band === 2).map((tick) => (
              <line
                key={tick.id}
                x1={tick.x1}
                y1={tick.y1}
                x2={tick.x2}
                y2={tick.y2}
                stroke="#8FF3EA"
                strokeWidth={tick.width}
                strokeOpacity={tick.opacity * 0.8}
                strokeLinecap="round"
              />
            ))}
          </Spin>

          <Spin dur={74 / speed} rev={false} play={play}>
            {MICRO_NODES.map((node) => (
              <rect
                key={node.id}
                x={node.cx - node.size / 2}
                y={node.cy - node.size / 2}
                width={node.size}
                height={node.size}
                fill="none"
                stroke="#A6FFF5"
                strokeWidth="0.7"
                strokeOpacity={node.opacity}
                transform={node.shape === 'diamond' ? `rotate(45 ${node.cx} ${node.cy})` : undefined}
              />
            ))}
          </Spin>
        </svg>

        {/* ============ 4. bloom + halo — ambient only, never pointer-driven */}
        <span className="orb-bloom cv-orb-motion" aria-hidden />
        <span className="orb-halo cv-orb-motion" aria-hidden />

        {/* ============ 5. glass core — its own hit target =========== */}
        <motion.div
          className={`orb-press${coreActive ? ' is-active' : ''}`}
          style={{ width: core, height: core, scaleX: pressX, scaleY: pressY, rotateX: tiltX, rotateY: tiltY }}
          onPointerEnter={() => setCoreActive(true)}
          onPointerMove={onCoreMove}
          onPointerLeave={onCoreLeave}
          onPointerDown={onCoreDown}
          onPointerUp={onCoreUp}
          onPointerCancel={onCoreLeave}
          onKeyDown={onCoreKeyDown}
          role="button"
          tabIndex={0}
          aria-label="Activate reliability core"
        >
          <div className="orb-core cv-orb-motion" ref={coreRef} aria-hidden>
            <motion.div className="orb-core-body" style={{ x: bodyX, y: bodyY }}>
              <svg className="orb-core-svg" viewBox={`0 0 ${BOX} ${BOX}`} preserveAspectRatio="none">
                <circle cx={CENTER} cy={CENTER} r={CENTER} fill={`url(#orbCore-${uid})`} />
              </svg>
              <span className="orb-core-grid" />
            </motion.div>

            <motion.span className="orb-core-falloff" style={{ x: falloffX, y: falloffY }} />
            <motion.span className="orb-core-sheen" style={{ x: specX, y: specY }} />
            <motion.span className="orb-core-rim" style={{ x: rimX, y: rimY }} />

            <motion.svg
              className="orb-core-particles"
              viewBox={`0 0 ${BOX} ${BOX}`}
              style={{ x: partX, y: partY, filter: particleGlow }}
            >
              <g filter={`url(#orbSoft-${uid})`}>
                {PARTICLES.filter((p) => p.zone === 'core').map((p) => (
                  <circle
                    key={p.id}
                    className={`${p.sparkle ? 'orb-sparkle' : 'orb-particle'} cv-orb-motion`}
                    cx={p.cx}
                    cy={p.cy}
                    r={p.r}
                    fill="#9BF0E7"
                    opacity={p.opacity}
                    style={{
                      ['--dur' as string]: `${p.duration.toFixed(1)}s`,
                      ['--dx' as string]: `${p.dx.toFixed(1)}px`,
                      ['--dy' as string]: `${p.dy.toFixed(1)}px`,
                      animationDelay: `${p.delay.toFixed(1)}s`,
                      animationPlayState: play,
                    }}
                  />
                ))}
              </g>
            </motion.svg>

            <svg className="orb-ripples" viewBox={`0 0 ${BOX} ${BOX}`}>
              {ripples.map((ripple) => (
                <motion.circle
                  key={ripple.key}
                  cx={ripple.cx}
                  cy={ripple.cy}
                  fill="none"
                  stroke="#BFFFF6"
                  initial={{ r: 4, opacity: 0.85, strokeWidth: 1.6 }}
                  animate={{ r: 58, opacity: 0, strokeWidth: 0.4 }}
                  transition={{ duration: 0.72, ease: [0.12, 0.7, 0.3, 1] }}
                />
              ))}
            </svg>
          </div>
        </motion.div>

        {/* ============ 6. travelling signal ========================= */}
        <motion.svg
          className="orb-ecg"
          viewBox="0 0 200 52"
          preserveAspectRatio="none"
          aria-hidden
          style={{ scaleY: ecgPulse }}
        >
          <defs>
            <linearGradient
              id={`ecgFade-${uid}`}
              x1="0"
              y1="0"
              x2={variant === 'stress' ? '100' : '1'}
              y2="0"
              gradientUnits={variant === 'stress' ? 'userSpaceOnUse' : undefined}
              spreadMethod={variant === 'stress' ? 'repeat' : undefined}
            >
              <stop offset="0%" stopColor="#8FF3EA" stopOpacity={variant === 'stress' ? 0.16 : 0} />
              <stop offset="14%" stopColor="#8FF3EA" stopOpacity={variant === 'stress' ? 0.22 : 0} />
              <stop offset="26%" stopColor="#8FF3EA" stopOpacity={variant === 'stress' ? 0.38 : 0.14} />
              <stop offset="38%" stopColor="#A9F8EF" stopOpacity={variant === 'stress' ? 0.68 : 0.6} />
              <stop offset="48%" stopColor="#E6FFFC" stopOpacity="1" />
              <stop offset="52%" stopColor="#E6FFFC" stopOpacity="1" />
              <stop offset="62%" stopColor="#A9F8EF" stopOpacity={variant === 'stress' ? 0.68 : 0.6} />
              <stop offset="74%" stopColor="#8FF3EA" stopOpacity={variant === 'stress' ? 0.38 : 0.14} />
              <stop offset="86%" stopColor="#8FF3EA" stopOpacity={variant === 'stress' ? 0.22 : 0} />
              <stop offset="100%" stopColor="#8FF3EA" stopOpacity={variant === 'stress' ? 0.16 : 0} />
            </linearGradient>
            <filter id={`orbEcgGlow-${uid}`} x="-20%" y="-300%" width="140%" height="700%">
              <feGaussianBlur stdDeviation="1.6" />
            </filter>
          </defs>
          <g
            ref={ecgTravelRef}
            className="orb-ecg-travel cv-orb-motion"
            style={{ animationPlayState: play }}
          >
            {(variant === 'stress' ? [0, 200] : [0]).map((offset) => (
              <path
                key={`glow-${offset}`}
                d={ECG_PATH}
                transform={offset ? `translate(${offset} 0)` : undefined}
                fill="none"
                stroke={`url(#ecgFade-${uid})`}
                strokeWidth="3.4"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.45"
                filter={`url(#orbEcgGlow-${uid})`}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {(variant === 'stress' ? [0, 200] : [0]).map((offset) => (
              <path
                key={`line-${offset}`}
                d={ECG_PATH}
                transform={offset ? `translate(${offset} 0)` : undefined}
                fill="none"
                stroke={`url(#ecgFade-${uid})`}
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
        </motion.svg>

        {/* ============ 7. central symbol + caption ================== */}
        <motion.div className="orb-symbol" style={{ x: symbolX, y: symbolY }} aria-hidden>
          <span className="orb-symbol-inner cv-orb-motion">
            <ReliabilityCoreIcon />
          </span>
        </motion.div>

        {caption && (
          <motion.div className="orb-caption" style={{ x: symbolX, y: symbolY }}>
            {caption}
          </motion.div>
        )}

        {/* ============ 8. front orbits + analyte connectors ========= */}
        <svg className="orb-svg orb-svg--front" viewBox={`0 0 ${BOX} ${BOX}`} aria-hidden>
          <defs>
            <filter id={`orbSoftF-${uid}`} x="-70%" y="-70%" width="240%" height="240%">
              <feGaussianBlur stdDeviation="1.6" />
            </filter>
          </defs>

          {frontOrbits.map((orbit) => (
            <OrbitLayer key={orbit.id} orbit={orbit} speed={speed} play={play} front {...orbitHover} />
          ))}
          <OrbitTravellers plane="front" uid={uid} speed={speed} play={play} hovered={hoveredOrbit} />

          {config.arcs &&
            ['M64 74 L77 87 L68 92 L86 110', 'M136 72 L123 85 L132 90 L114 108'].map((d, i) => (
              <path
                key={d}
                className="orb-arc cv-orb-motion"
                d={d}
                fill="none"
                stroke="#C8FFF7"
                strokeWidth="1"
                strokeLinecap="round"
                style={{
                  ['--dur' as string]: `${(2.9 + i * 1.1) / speed}s`,
                  animationDelay: `${-i * 0.8}s`,
                  animationPlayState: play,
                }}
              />
            ))}

          {/* analyte connectors — driven only by their own chip's hover */}
          {nodes.map((node, i) => {
            const nx = node.x * BOX
            const ny = node.y * BOX
            const dx = CENTER - nx
            const dy = CENTER - ny
            const len = Math.hypot(dx, dy) || 1
            const ux = dx / len
            const uy = dy / len
            const x1 = nx + ux * 14
            const y1 = ny + uy * 14
            const x2 = nx + ux * (len - 44)
            const y2 = ny + uy * (len - 44)
            const bow = [7.5, -5.5, 3.5][i % 3]
            const mx = (x1 + x2) / 2 - uy * bow
            const my = (y1 + y2) / 2 + ux * bow
            const d = `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`
            const on = hoveredAnalyte === node.id
            return (
              <g key={node.id} className="orb-trail">
                <path
                  d={d}
                  fill="none"
                  stroke={on ? '#A6FFF5' : '#5FE0D6'}
                  strokeWidth={on ? 0.75 : 0.5}
                  strokeOpacity={on ? 0.6 : 0.28}
                  strokeLinecap="round"
                />
                <path
                  className="orb-pulse cv-orb-motion"
                  d={d}
                  fill="none"
                  stroke="#EAFFFC"
                  strokeWidth={on ? 1.5 : 1.1}
                  strokeLinecap="round"
                  filter={`url(#orbSoftF-${uid})`}
                  style={{
                    ['--dur' as string]: `${(2.9 / speed).toFixed(2)}s`,
                    animationDelay: `${-(i * 0.9).toFixed(1)}s`,
                    animationPlayState: play,
                  }}
                />
                <path
                  className="orb-pulse cv-orb-motion"
                  d={d}
                  fill="none"
                  stroke="#FFFFFF"
                  strokeWidth={on ? 0.8 : 0.55}
                  strokeLinecap="round"
                  style={{
                    ['--dur' as string]: `${(2.9 / speed).toFixed(2)}s`,
                    animationDelay: `${-(i * 0.9).toFixed(1)}s`,
                    animationPlayState: play,
                  }}
                />
                {/* the core entry point brightens with its own analyte */}
                <circle
                  cx={x2}
                  cy={y2}
                  r={on ? 2.2 : 1.2}
                  fill={on ? '#EAFFFC' : '#5FE0D6'}
                  opacity={on ? 1 : 0.45}
                  style={{ transition: 'r 220ms ease, opacity 220ms ease' }}
                />
              </g>
            )
          })}
        </svg>
      </div>

      {/* ============ 9. analyte chips — each its own hit target ==== */}
      {nodes.length > 0 && (
        <div className="orb-nodes">
          {nodes.map((node, i) => (
            <div
              key={node.id}
              className={`orb-chip cv-orb-motion${hoveredAnalyte === node.id ? ' is-hovered' : ''}`}
              style={{
                left: `${node.x * 100}%`,
                top: `${node.y * 100}%`,
                width: node.width,
                ['--dur' as string]: `${(5.2 + i * 0.7).toFixed(1)}s`,
                animationDelay: `${-i * 1.4}s`,
                animationPlayState: play,
              }}
              onPointerEnter={() => setHoveredAnalyte(node.id)}
              onPointerLeave={() => setHoveredAnalyte(null)}
              onFocus={() => setHoveredAnalyte(node.id)}
              onBlur={() => setHoveredAnalyte(null)}
              tabIndex={0}
              role="img"
              aria-label={`${node.label}${node.sub ? `, ${node.sub}` : ''}`}
              title={node.sub}
            >
              <span className="orb-chip-icon" aria-hidden>
                {node.icon}
              </span>
              <span className="orb-chip-text">
                <strong>{node.label}</strong>
                {node.sub && <em>{node.sub}</em>}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ---------------------------------------------------------------------- */
/* building blocks                                                         */
/* ---------------------------------------------------------------------- */

/** A continuously rotating group. Rotation is ambient — never pointer-driven. */
function Spin({
  dur,
  rev,
  play,
  children,
  className = '',
}: {
  dur: number
  rev: boolean
  play: string
  children: ReactNode
  className?: string
}) {
  return (
    <g
      className={`orb-spin cv-orb-motion ${className}`}
      style={{
        ['--dur' as string]: `${dur.toFixed(1)}s`,
        animationDirection: rev ? 'reverse' : 'normal',
        animationPlayState: play,
        transformOrigin: `${CENTER}px ${CENTER}px`,
      }}
    >
      {children}
    </g>
  )
}

interface HoverProps {
  hovered: string | null
  onEnter: (id: string) => void
  onLeave: () => void
  reduced: boolean
}

/**
 * One technical band. It carries a wide transparent hit stroke so a 1px path
 * stays easy to target, and reacts only when it is itself hovered.
 */
function RingLayer({
  ring,
  uid,
  speed,
  play,
  hovered,
  onEnter,
  onLeave,
  reduced,
}: { ring: Ring; uid: string; speed: number; play: string } & HoverProps) {
  const on = hovered === ring.id
  const d = ring.arc ? arcPath(ring.r, ring.arc[0], ring.arc[1]) : undefined
  // a hovered band accelerates slightly and drifts out by ~1.5px
  const dur = (Math.abs(ring.spin) / speed) * (on && !reduced ? 0.72 : 1)

  const shared = {
    fill: 'none',
    stroke: ring.stroke,
    strokeWidth: ring.width * (on ? 1.5 : 1),
    strokeOpacity: Math.min(1, ring.opacity * (on ? 1.6 : 1)),
    strokeDasharray: ring.dash,
    strokeLinecap: 'round' as const,
    filter: ring.blur ? `url(#orbBloom-${uid})` : undefined,
    style: { transition: 'stroke-opacity 240ms ease, stroke-width 240ms ease' },
  }

  return (
    <g
      className="orb-band"
      style={{
        transform: on && !reduced ? 'scale(1.012)' : 'scale(1)',
        transformOrigin: `${CENTER}px ${CENTER}px`,
        transition: 'transform 320ms cubic-bezier(0.16,1,0.3,1)',
      }}
    >
      <Spin dur={dur} rev={ring.spin < 0} play={play}>
        {d ? <path d={d} {...shared} /> : <circle cx={CENTER} cy={CENTER} r={ring.r} {...shared} />}
        {/* one pulse runs the band while it is hovered */}
        {on && !reduced && (
          <path
            className="orb-band-pulse"
            d={d ?? arcPath(ring.r, -179.9, 179.9)}
            fill="none"
            stroke="#EAFFFC"
            strokeWidth={ring.width * 1.7}
            strokeLinecap="round"
          />
        )}
        {/* transparent hit stroke: visually precise, comfortably hoverable */}
        {d ? (
          <path
            d={d}
            fill="none"
            stroke="transparent"
            strokeWidth={HIT_STROKE}
            className="orb-hit"
            onPointerEnter={() => onEnter(ring.id)}
            onPointerLeave={onLeave}
          />
        ) : (
          <circle
            cx={CENTER}
            cy={CENTER}
            r={ring.r}
            fill="none"
            stroke="transparent"
            strokeWidth={HIT_STROKE}
            className="orb-hit"
            onPointerEnter={() => onEnter(ring.id)}
            onPointerLeave={onLeave}
          />
        )}
      </Spin>
    </g>
  )
}

function OrbitLayer({
  orbit,
  speed,
  play,
  front,
  hovered,
  onEnter,
  onLeave,
  reduced,
}: { orbit: Orbit; speed: number; play: string; front?: boolean } & HoverProps) {
  const on = hovered === orbit.id
  const dur = (Math.abs(orbit.spin) / speed) * (on && !reduced ? 0.72 : 1)
  const geom = {
    cx: CENTER,
    cy: CENTER,
    rx: orbit.rx,
    ry: orbit.ry,
    transform: `rotate(${orbit.rotate} ${CENTER} ${CENTER})`,
  }

  return (
    <g
      style={{
        transform: on && !reduced ? 'scale(1.012)' : 'scale(1)',
        transformOrigin: `${CENTER}px ${CENTER}px`,
        transition: 'transform 320ms cubic-bezier(0.16,1,0.3,1)',
      }}
    >
      <Spin dur={dur} rev={orbit.spin < 0} play={play}>
        <ellipse
          {...geom}
          fill="none"
          stroke={front ? '#8FF3EA' : 'url(#orbRingGrad)'}
          strokeWidth={(front ? 0.85 : 0.7) * (on ? 1.6 : 1)}
          strokeOpacity={Math.min(
            1,
            (front ? orbit.opacity * 1.25 : orbit.opacity * 0.7) * (on ? 1.9 : 1),
          )}
          strokeDasharray={orbit.dash}
          strokeLinecap="round"
          style={{ transition: 'stroke-opacity 240ms ease, stroke-width 240ms ease' }}
        />
        <ellipse
          {...geom}
          fill="none"
          stroke="transparent"
          strokeWidth={HIT_STROKE}
          className="orb-hit"
          onPointerEnter={() => onEnter(orbit.id)}
          onPointerLeave={onLeave}
        />
      </Spin>
    </g>
  )
}

/**
 * Travellers are grouped per orbit so the ellipse rotation lives on a parent
 * `<g>`. `offset-path` composes after the element's own transform, so rotating
 * the traveller directly would spin the dot instead of tilting its path.
 */
function OrbitTravellers({
  plane,
  uid,
  speed,
  play,
  hovered,
}: {
  plane: 'rear' | 'front'
  uid: string
  speed: number
  play: string
  hovered: string | null
}) {
  const orbits = ORBITS.filter((orbit) => orbit.plane === plane)
  return (
    <g
      filter={plane === 'rear' ? `url(#orbRearSoft-${uid})` : undefined}
      opacity={plane === 'rear' ? 0.72 : 1}
    >
      {orbits.map((orbit) => (
        <g
          key={orbit.id}
          transform={`rotate(${orbit.rotate} ${CENTER} ${CENTER})`}
          // nodes brighten only when their own orbit is the hovered one
          opacity={hovered === orbit.id ? 1 : 0.82}
          style={{ transition: 'opacity 240ms ease' }}
        >
          {ORBIT_NODES.filter((node) => node.orbit.id === orbit.id).map((node) => (
            <TravellerNode
              key={node.id}
              node={node}
              orbit={orbit}
              uid={uid}
              speed={speed}
              play={play}
              active={hovered === orbit.id}
            />
          ))}
        </g>
      ))}
    </g>
  )
}

function TravellerNode({
  node,
  orbit,
  uid,
  speed,
  play,
  active,
}: {
  node: OrbitNode
  orbit: Orbit
  uid: string
  speed: number
  play: string
  active: boolean
}) {
  const style = {
    offsetPath: `path("${ellipsePath(orbit.rx, orbit.ry)}")`,
    offsetDistance: `${node.offset}%`,
    ['--dur' as string]: `${(node.duration / speed).toFixed(1)}s`,
    animationDelay: `${-(node.offset / 4).toFixed(1)}s`,
    animationPlayState: play,
  } as CSSProperties
  const r = node.r * (active ? 1.35 : 1)
  const opacity = Math.min(1, node.opacity * (active ? 1.5 : 1))

  if (node.tier === 'special') {
    return (
      <g className="orb-traveller cv-orb-motion" style={style}>
        <circle className="orb-node-halo cv-orb-motion" r={r * 2.6} fill="#5FF0E2" opacity="0.16" />
        <circle r={r * 1.75} fill="none" stroke="#BFFFF6" strokeWidth="0.5" opacity="0.75" />
        <circle r={r} fill={`url(#orbNode-${uid})`} />
        <circle r={r * 0.36} fill="#FFFFFF" />
      </g>
    )
  }

  if (node.tier === 'bright') {
    return (
      <g className="orb-traveller cv-orb-motion" style={style}>
        <circle r={r} fill={`url(#orbNode-${uid})`} opacity={opacity} />
        <circle r={r * 0.34} fill="#EAFFFC" />
      </g>
    )
  }

  return (
    <circle
      className="orb-traveller cv-orb-motion"
      r={r}
      fill={node.tier === 'medium' ? '#6FE3DA' : '#4FC7C0'}
      opacity={opacity}
      style={style}
    />
  )
}
