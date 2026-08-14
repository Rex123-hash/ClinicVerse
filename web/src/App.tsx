import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import AppShell from './components/shell/AppShell'
import PageSkeleton from './components/ui/PageSkeleton'

const Overview = lazy(() => import('./pages/Overview'))
const ModelLab = lazy(() => import('./pages/ModelLab'))
const StressLab = lazy(() => import('./pages/StressLab'))
const ReliabilityReport = lazy(() => import('./pages/ReliabilityReport'))
const Experiments = lazy(() => import('./pages/Experiments'))
const Artifacts = lazy(() => import('./pages/Artifacts'))
const NotFound = lazy(() => import('./pages/NotFound'))

export default function App() {
  const location = useLocation()

  return (
    <AppShell>
      <Suspense fallback={<PageSkeleton />}>
        <AnimatePresence mode="wait" initial={false}>
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<Overview />} />
            <Route path="/model-lab" element={<ModelLab />} />
            <Route path="/stress-lab" element={<StressLab />} />
            <Route path="/reliability-report" element={<ReliabilityReport />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/artifacts" element={<Artifacts />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </AppShell>
  )
}
