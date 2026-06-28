import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Leaf, BarChart2, Info } from 'lucide-react'

const links = [
  { to: '/', label: 'Detect', icon: Leaf },
  { to: '/history', label: 'History', icon: BarChart2 },
  { to: '/about', label: 'About', icon: Info },
]

export default function MobileNav() {
  const { pathname } = useLocation()

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-t border-slate-200/60 pb-[env(safe-area-inset-bottom)]">
      <div className="flex items-center justify-around h-16">
        {links.map(({ to, label, icon: Icon }) => {
          const active = pathname === to
          return (
            <Link key={to} to={to} className="relative flex flex-col items-center gap-1 py-2 px-6">
              {active && (
                <motion.div
                  layoutId="mob-nav"
                  className="absolute -top-px left-1/2 -translate-x-1/2 w-10 h-1 rounded-full bg-blue-500"
                  transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                />
              )}
              <motion.div animate={active ? { scale: 1.2, y: -2 } : { scale: 1, y: 0 }} transition={{ type: 'spring' }}>
                <Icon className={`w-5 h-5 ${active ? 'text-blue-600' : 'text-slate-400'}`} />
              </motion.div>
              <span className={`text-[10px] font-bold ${active ? 'text-blue-600' : 'text-slate-400'}`}>{label}</span>
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
