/**
 * Deterministic geometry for the Cliniverse orb.
 *
 * Everything here is computed once from a fixed seed so particles, orbital
 * nodes and sparkles keep identical positions across every React render and
 * every reload. Nothing in the orb may reshuffle on re-render.
 *
 * All coordinates live in a 0–200 viewBox with the core centred at (100, 100).
 */

export const BOX = 200
export const CENTER = 100

/** Mulberry32 — small, fast, fully deterministic. */
function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/* ---------------------------------------------------------------------- */
/* concentric ring system — 7 visually distinct layers                     */
/* ---------------------------------------------------------------------- */

export interface Ring {
  id: string
  r: number
  stroke: string
  width: number
  opacity: number
  dash?: string
  /** seconds; negative spins counter-clockwise */
  spin: number
  /** parallax depth multiplier */
  depth: number
  blur?: boolean
  /**
   * Degrees [start, end]. When present the band is drawn as a partial arc
   * rather than a closed circle — instrument dials, not concentric circles.
   */
  arc?: [number, number]
}

export const RINGS: readonly Ring[] = [
  // 1. inner luminous rim, tight to the glass core — the one deliberate circle
  {
    id: 'rim',
    r: 51.5,
    stroke: 'url(#orbRimGrad)',
    width: 1.5,
    opacity: 1,
    spin: 31,
    depth: 0.9,
  },
  // 2. thin turquoise ring, broken into two opposed arcs
  {
    id: 'thin-a',
    r: 56,
    stroke: '#5FF0E2',
    width: 0.8,
    opacity: 0.72,
    spin: -24,
    depth: 0.95,
    arc: [-142, 34],
  },
  {
    id: 'thin-b',
    r: 56,
    stroke: '#5FF0E2',
    width: 0.8,
    opacity: 0.44,
    spin: -24,
    depth: 0.95,
    arc: [64, 148],
  },
  // 3. translucent aqua band — the only blurred ring, kept narrow
  {
    id: 'band',
    r: 62,
    stroke: 'url(#orbHaloGrad)',
    width: 5,
    opacity: 0.3,
    spin: 38,
    depth: 1,
    blur: true,
    arc: [-108, 96],
  },
  // 4. segmented technical band with controlled gaps
  {
    id: 'segment',
    r: 70,
    stroke: 'url(#orbRingGrad)',
    width: 1.5,
    opacity: 0.96,
    dash: '26 9 3 9 8 9',
    spin: -31,
    depth: 1.05,
  },
  // 5. dotted measurement ring, three-quarter sweep
  {
    id: 'dotted',
    r: 78,
    stroke: '#7DE8DE',
    width: 1.2,
    opacity: 0.66,
    dash: '0.7 6.5',
    spin: 24,
    depth: 1.1,
    arc: [-160, 96],
  },
  // 6. outer thin cyan ring
  {
    id: 'outer',
    r: 87,
    stroke: 'url(#orbRingGrad)',
    width: 0.9,
    opacity: 0.74,
    dash: '44 12 3 12',
    spin: -38,
    depth: 1.15,
  },
  // 7. faint atmospheric arc, deliberately incomplete
  {
    id: 'atmos',
    r: 95,
    stroke: '#43D6CC',
    width: 0.7,
    opacity: 0.34,
    dash: '1.3 9',
    spin: 62,
    depth: 1.2,
    arc: [-52, 172],
  },
]

/** Short bright accent arcs — the focal detail on the technical bands. */
export interface AccentArc {
  id: string
  r: number
  arc: [number, number]
  width: number
  opacity: number
  spin: number
  depth: number
}

export const ACCENT_ARCS: readonly AccentArc[] = [
  { id: 'a1', r: 70, arc: [-96, -58], width: 2.1, opacity: 0.92, spin: -31, depth: 1.05 },
  { id: 'a2', r: 78, arc: [22, 44], width: 1.7, opacity: 0.72, spin: 24, depth: 1.1 },
  { id: 'a3', r: 87, arc: [136, 178], width: 1.5, opacity: 0.62, spin: -38, depth: 1.15 },
  { id: 'a4', r: 56, arc: [168, 190], width: 1.4, opacity: 0.66, spin: -24, depth: 0.95 },
]

/**
 * Radial ticks on three bands. Most sit at low opacity; every fifth is a major
 * graduation, which is what makes the band read as an instrument scale.
 */
export interface Tick {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  width: number
  opacity: number
  band: number
}

export function buildTicks(): readonly Tick[] {
  const bands = [
    { r: 66, count: 48, len: 2.2, spin: 0 },
    { r: 74.5, count: 32, len: 3, spin: 0 },
    { r: 91, count: 24, len: 2.6, spin: 0 },
  ]
  const ticks: Tick[] = []
  bands.forEach((band, bi) => {
    for (let i = 0; i < band.count; i += 1) {
      const major = i % 5 === 0
      const angle = (i / band.count) * Math.PI * 2
      const len = major ? band.len * 1.7 : band.len
      const cos = Math.cos(angle)
      const sin = Math.sin(angle)
      ticks.push({
        id: `t${bi}-${i}`,
        x1: CENTER + cos * band.r,
        y1: CENTER + sin * band.r,
        x2: CENTER + cos * (band.r + len),
        y2: CENTER + sin * (band.r + len),
        width: major ? 0.95 : 0.6,
        opacity: major ? 0.8 : 0.32,
        band: bi,
      })
    }
  })
  return ticks
}

export const TICKS = buildTicks()

/** A handful of square/diamond micro-nodes pinned to the bands. */
export interface MicroNode {
  id: string
  cx: number
  cy: number
  size: number
  opacity: number
  shape: 'diamond' | 'square'
}

export function buildMicroNodes(): readonly MicroNode[] {
  const spec: [number, number, 'diamond' | 'square', number][] = [
    [70, -68, 'diamond', 0.85],
    [70, 118, 'square', 0.45],
    [78, -14, 'square', 0.5],
    [87, 62, 'diamond', 0.7],
    [56, -160, 'diamond', 0.4],
  ]
  return spec.map(([r, deg, shape, opacity], i) => {
    const rad = (deg * Math.PI) / 180
    return {
      id: `m${i}`,
      cx: CENTER + Math.cos(rad) * r,
      cy: CENTER + Math.sin(rad) * r,
      size: shape === 'diamond' ? 2.4 : 1.9,
      opacity,
      shape,
    }
  })
}

export const MICRO_NODES = buildMicroNodes()

/** Partial-arc path on a circle of radius `r`, from `a0` to `a1` degrees. */
export function arcPath(r: number, a0: number, a1: number): string {
  const rad = (a: number) => (a * Math.PI) / 180
  const x0 = CENTER + r * Math.cos(rad(a0))
  const y0 = CENTER + r * Math.sin(rad(a0))
  const x1 = CENTER + r * Math.cos(rad(a1))
  const y1 = CENTER + r * Math.sin(rad(a1))
  const large = Math.abs(a1 - a0) > 180 ? 1 : 0
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`
}

/* ---------------------------------------------------------------------- */
/* elliptical orbits                                                        */
/* ---------------------------------------------------------------------- */

export interface Orbit {
  id: string
  rx: number
  ry: number
  rotate: number
  /** rear orbits render behind the core, front orbits in front of it */
  plane: 'rear' | 'front'
  opacity: number
  dash: string
  spin: number
  depth: number
}

export const ORBITS: readonly Orbit[] = [
  { id: 'e1', rx: 97, ry: 50, rotate: -17, plane: 'rear', opacity: 0.42, dash: '5 6', spin: 46, depth: 0.72 },
  { id: 'e2', rx: 90, ry: 63, rotate: 27, plane: 'rear', opacity: 0.3, dash: '2 7', spin: -58, depth: 0.76 },
  { id: 'e3', rx: 68, ry: 96, rotate: 9, plane: 'front', opacity: 0.34, dash: '4 8', spin: 52, depth: 1.35 },
  { id: 'e4', rx: 98, ry: 36, rotate: 63, plane: 'front', opacity: 0.26, dash: '2.5 9', spin: -66, depth: 1.4 },
]

/** SVG path for an ellipse, so nodes can travel it with offset-path. */
export function ellipsePath(rx: number, ry: number): string {
  return `M ${CENTER - rx} ${CENTER} a ${rx} ${ry} 0 1 0 ${rx * 2} 0 a ${rx} ${ry} 0 1 0 ${-rx * 2} 0`
}

/* ---------------------------------------------------------------------- */
/* orbital nodes travelling the ellipses                                    */
/* ---------------------------------------------------------------------- */

/**
 * Node tiers, so the orbits read as having focal points rather than being
 * evenly decorated. Distribution across the 22 nodes is roughly
 * 60% faint / 25% medium / 10% bright / 5% special.
 */
export type NodeTier = 'faint' | 'medium' | 'bright' | 'special'

const TIER_SEQUENCE: readonly NodeTier[] = [
  // 13 faint, 6 medium, 2 bright, 1 special — interleaved so tiers do not clump
  'faint', 'faint', 'medium', 'faint', 'bright', 'faint', 'faint',
  'medium', 'faint', 'faint', 'special', 'faint', 'medium', 'faint',
  'faint', 'medium', 'bright', 'faint', 'medium', 'faint', 'faint', 'medium',
]

const TIER_STYLE: Record<NodeTier, { r: number; opacity: number }> = {
  faint: { r: 0.85, opacity: 0.34 },
  medium: { r: 1.5, opacity: 0.62 },
  bright: { r: 2.4, opacity: 0.95 },
  special: { r: 2.8, opacity: 1 },
}

export interface OrbitNode {
  id: string
  orbit: Orbit
  r: number
  offset: number
  duration: number
  opacity: number
  tier: NodeTier
}

export function buildOrbitNodes(seed = 20260809): readonly OrbitNode[] {
  const rand = rng(seed)
  const nodes: OrbitNode[] = []
  const perOrbit = [7, 6, 5, 4]
  let index = 0

  ORBITS.forEach((orbit, oi) => {
    const count = perOrbit[oi]
    for (let i = 0; i < count; i += 1) {
      const tier = TIER_SEQUENCE[index % TIER_SEQUENCE.length]
      const style = TIER_STYLE[tier]
      // foreground orbits carry slightly larger, brighter nodes
      const foreground = orbit.plane === 'front' ? 1.2 : 1
      nodes.push({
        id: `${orbit.id}-${i}`,
        orbit,
        r: style.r * foreground * (0.88 + rand() * 0.24),
        offset: (i / count) * 100 + rand() * 6,
        duration: 26 + rand() * 26,
        opacity: Math.min(1, style.opacity * (orbit.plane === 'front' ? 1.1 : 0.82)),
        tier,
      })
      index += 1
    }
  })
  return nodes
}

/* ---------------------------------------------------------------------- */
/* atmospheric particles                                                    */
/* ---------------------------------------------------------------------- */

export interface Particle {
  id: number
  cx: number
  cy: number
  r: number
  opacity: number
  duration: number
  delay: number
  dx: number
  dy: number
  sparkle: boolean
  /** inside the core, or in the atmosphere around the rings */
  zone: 'core' | 'atmos'
}

export function buildParticles(seed = 424242): readonly Particle[] {
  const rand = rng(seed)
  const particles: Particle[] = []

  // 14 faint particles inside the glass core
  for (let i = 0; i < 14; i += 1) {
    const angle = rand() * Math.PI * 2
    const radius = rand() * 40
    particles.push({
      id: i,
      cx: CENTER + Math.cos(angle) * radius,
      cy: CENTER + Math.sin(angle) * radius,
      r: 0.4 + rand() * 0.7,
      opacity: 0.2 + rand() * 0.45,
      duration: 11 + rand() * 12,
      delay: -rand() * 14,
      dx: (rand() - 0.5) * 7,
      dy: (rand() - 0.5) * 7,
      sparkle: rand() > 0.84,
      zone: 'core',
    })
  }

  // 22 atmospheric particles drifting between the rings
  for (let i = 0; i < 22; i += 1) {
    const angle = rand() * Math.PI * 2
    const radius = 54 + rand() * 46
    particles.push({
      id: 100 + i,
      cx: CENTER + Math.cos(angle) * radius,
      cy: CENTER + Math.sin(angle) * radius * 0.92,
      r: 0.45 + rand() * 1.0,
      opacity: 0.25 + rand() * 0.5,
      duration: 13 + rand() * 16,
      delay: -rand() * 18,
      dx: (rand() - 0.5) * 13,
      dy: (rand() - 0.5) * 13,
      sparkle: rand() > 0.78,
      zone: 'atmos',
    })
  }

  return particles
}

/* ---------------------------------------------------------------------- */
/* telemetry waveform                                                       */
/* ---------------------------------------------------------------------- */

/**
 * One period of the signal, drawn entirely in cubic segments so there is not a
 * single hard corner in it.
 *
 * Morphology, left to right: calm baseline → soft low oscillation → narrow
 * downward deflection → sharp central peak → controlled recovery overshoot →
 * secondary low-amplitude wave → calm baseline.
 *
 * The period spans 100 units on a 52-unit canvas with the baseline at y = 26,
 * so two periods laid end to end tile seamlessly under a -50% translation.
 */
function ecgPeriod(x: number): string {
  const p = (n: number) => (n + x).toFixed(2)
  return [
    // calm baseline
    `C ${p(5)} 26, ${p(9)} 26, ${p(13)} 26`,
    // small soft oscillation
    `C ${p(16.5)} 26, ${p(17.5)} 23.2, ${p(20.5)} 23.2`,
    `C ${p(23.5)} 23.2, ${p(24.5)} 27.6, ${p(27.5)} 27.6`,
    `C ${p(30)} 27.6, ${p(31)} 26, ${p(33.5)} 26`,
    // narrow downward deflection
    `C ${p(35.5)} 26, ${p(36.4)} 31.4, ${p(38.6)} 31.4`,
    // sharp but elegant central peak
    `C ${p(40.2)} 31.4, ${p(40.8)} 7.4, ${p(43.6)} 7.4`,
    // controlled recovery with a single restrained overshoot
    `C ${p(46.4)} 7.4, ${p(46.9)} 33.6, ${p(49.4)} 33.6`,
    `C ${p(51.6)} 33.6, ${p(52.6)} 26, ${p(55.5)} 26`,
    `C ${p(58.5)} 26, ${p(60.5)} 26, ${p(63)} 26`,
    // secondary low-amplitude wave
    `C ${p(66)} 26, ${p(67.4)} 22.6, ${p(70.4)} 22.6`,
    `C ${p(73.4)} 22.6, ${p(74.8)} 27.4, ${p(77.8)} 27.4`,
    `C ${p(80.8)} 27.4, ${p(82.4)} 26, ${p(85)} 26`,
    // calm baseline out
    `C ${p(90)} 26, ${p(95)} 26, ${p(100)} 26`,
  ].join(' ')
}

export const ECG_PATH = `M 0 26 ${ecgPeriod(0)} ${ecgPeriod(100)}`
