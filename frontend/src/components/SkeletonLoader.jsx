import { motion } from 'framer-motion'
import { Leaf } from 'lucide-react'

export function LeafSpinner() {
  return (
    <div className="flex flex-col items-center gap-4">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        className="w-16 h-16 rounded-full border-4 border-blue-100 border-t-blue-500 flex items-center justify-center"
      >
        <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
          <Leaf className="w-6 h-6 text-blue-500" />
        </motion.div>
      </motion.div>
      <div className="flex gap-1.5">
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1, 0.8] }}
            transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
            className="w-2 h-2 rounded-full bg-blue-400"
          />
        ))}
      </div>
      <p className="text-sm text-slate-400 font-medium">Analyzing your leaf...</p>
    </div>
  )
}

export function SkeletonCard() {
  return (
    <div className="space-y-3">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl skeleton" />
          <div className="flex-1 space-y-2">
            <div className="h-5 w-48 skeleton" />
            <div className="h-3 w-24 skeleton" />
          </div>
        </div>
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="flex items-center gap-3">
              <div className="h-2.5 w-24 skeleton" />
              <div className="flex-1 h-2.5 skeleton" />
              <div className="h-2.5 w-8 skeleton" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
