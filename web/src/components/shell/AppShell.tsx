import { useState, type ReactNode } from 'react'
import Sidebar from './Sidebar'
import TopHeader from './TopHeader'

export default function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="cv-shell" data-collapsed={collapsed || undefined}>
      <a className="cv-skip" href="#cv-main">
        Skip to main content
      </a>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <div className="cv-shell-body">
        <TopHeader />
        <main className="cv-main" id="cv-main" tabIndex={-1}>
          {children}
        </main>
      </div>
    </div>
  )
}
