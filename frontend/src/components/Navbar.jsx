import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Leaf, BarChart2, Info, Dna } from 'lucide-react'

const links = [
  { to: '/', label: 'Detect', icon: Leaf },
  { to: '/history', label: 'History', icon: BarChart2 },
  { to: '/about', label: 'About', icon: Info },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav className="sticky top-0 z-50 bg-white/70 backdrop-blur-xl border-b border-slate-200/60">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group">
          <motion.div
            whileHover={{ rotate: 180, scale: 1.1 }}
            transition={{ duration: 0.5 }}
            className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20"
          >
            <Dna className="w-5 h-5 text-white" />
          </motion.div>
          <span className="text-xl font-black tracking-tight gradient-text">CassavaAI</span>
        </Link>

        <div className="flex items-center gap-1 bg-slate-100 rounded-full p-1">
          {links.map(({ to, label, icon: Icon }) => {
            const active = pathname === to
            return (
              <Link key={to} to={to} className="relative px-5 py-2 rounded-full text-sm font-medium transition-all">
                {active && (
                  <motion.div
                    layoutId="nav-bg"
                    className="absolute inset-0 bg-white rounded-full shadow-md shadow-blue-500/10"
                    transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                  />
                )}
                <span className={`relative z-10 flex items-center gap-1.5 ${active ? 'text-blue-600' : 'text-slate-500 hover:text-blue-500'}`}>
                  <Icon className="w-4 h-4" />
                  {label}
                </span>
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
