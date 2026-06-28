import { motion } from 'framer-motion'

const COLORS = {
  mosaic_disease:       { bar: 'from-amber-400 to-yellow-500', text: 'text-amber-600' },
  bacterial_blight:     { bar: 'from-red-400 to-rose-500', text: 'text-red-600' },
  brown_streak_disease: { bar: 'from-orange-400 to-amber-500', text: 'text-orange-600' },
  green_mottle:         { bar: 'from-teal-400 to-emerald-500', text: 'text-teal-600' },
  healthy:              { bar: 'from-blue-400 to-indigo-500', text: 'text-blue-600' },
}

export default function ConfidenceBar({ name, confidence, isTop, delay = 0 }) {
  const { bar, text } = COLORS[name] || COLORS.healthy
  const pct = Math.round(confidence * 100)

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: delay * 0.08, duration: 0.4 }}
      className="flex items-center gap-3"
    >
      <span className={`text-xs font-medium w-28 text-right truncate capitalize ${isTop ? text + ' font-bold' : 'text-slate-400'}`}>
        {name.replace(/_/g, ' ')}
      </span>
      <div className="flex-1 bg-slate-100 rounded-full h-2.5 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, delay: delay * 0.08 + 0.2, ease: [0.22, 1, 0.36, 1] }}
          className={`h-full rounded-full bg-gradient-to-r ${bar}`}
        />
      </div>
      <span className={`text-xs font-mono w-10 ${isTop ? text + ' font-bold' : 'text-slate-400'}`}>{pct}%</span>
    </motion.div>
  )
}
