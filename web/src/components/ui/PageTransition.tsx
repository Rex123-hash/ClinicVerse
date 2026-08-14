import { motion } from 'framer-motion'
import { useEffect, type ReactNode } from 'react'

/** Route-level fade + lift. The shell stays put; only page content moves. */
export default function PageTransition({
  children,
  title,
}: {
  children: ReactNode
  title: string
}) {
  useEffect(() => {
    document.title = `${title} · Cliniverse`
  }, [title])

  return (
    <motion.div
      className="cv-page"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.36, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  )
}
