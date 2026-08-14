import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Activity,
  Box,
  ChevronLeft,
  FlaskConical,
  Home,
  Layers,
  LineChart,
  ShieldCheck,
} from 'lucide-react'
import CliniverseMark from '../brand/CliniverseMark'

const NAV = [
  { to: '/overview', label: 'Overview', Icon: Home },
  { to: '/model-lab', label: 'Model Lab', Icon: Box },
  { to: '/stress-lab', label: 'Stress Lab', Icon: Activity },
  { to: '/reliability-report', label: 'Reliability Report', Icon: LineChart },
  { to: '/experiments', label: 'Experiments', Icon: FlaskConical },
  { to: '/artifacts', label: 'Artifacts', Icon: Layers },
] as const

interface Props {
  collapsed: boolean
  onToggle: () => void
}

export default function Sidebar({ collapsed, onToggle }: Props) {
  return (
    <aside className="cv-sidebar" aria-label="Primary">
      <div className="cv-sidebar-mark">
        <CliniverseMark size={28} />
      </div>

      <nav className="cv-nav">
        {NAV.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `cv-nav-item${isActive ? ' is-active' : ''}`}
            title={collapsed ? label : undefined}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="cv-nav-active"
                    className="cv-nav-active-bg"
                    transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                  />
                )}
                <Icon size={17} strokeWidth={1.75} className="cv-nav-icon" aria-hidden />
                <span className="cv-nav-label">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="cv-sidebar-foot">
        <div className="cv-status-card">
          <span className="cv-status-icon" aria-hidden>
            <ShieldCheck size={15} strokeWidth={1.9} />
          </span>
          <span className="cv-status-text">
            <strong>Evidence status</strong>
            <em>
              Committed artifacts loaded
              <i className="cv-live-dot" aria-hidden />
            </em>
          </span>
        </div>

        <button type="button" className="cv-collapse" onClick={onToggle}>
          <ChevronLeft size={15} strokeWidth={1.9} className="cv-collapse-chevron" aria-hidden />
          <span>{collapsed ? 'Expand' : 'Collapse'}</span>
        </button>
      </div>
    </aside>
  )
}
