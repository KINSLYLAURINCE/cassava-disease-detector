import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, AlertTriangle, ExternalLink, ChevronDown, ChevronUp, XCircle, Shield } from 'lucide-react'
import { useState } from 'react'
import ConfidenceBar from './ConfidenceBar'
import FeedbackPanel from './FeedbackPanel'

const DISEASE_STYLE = {
  healthy:              { emoji: '🌿', bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700' },
  mosaic_disease:       { emoji: '🟡', bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700' },
  bacterial_blight:     { emoji: '🔴', bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700' },
  brown_streak_disease: { emoji: '🟠', bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700' },
  green_mottle:         { emoji: '🟢', bg: 'bg-teal-50', border: 'border-teal-200', text: 'text-teal-700' },
}

function Section({ title, icon: Icon, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-slate-600 hover:text-blue-600 transition-colors"
      >
        <span className="flex items-center gap-2"><Icon className="w-4 h-4 text-blue-500" />{title}</span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 border-t border-slate-100 pt-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function ResultCard({ result, imageFile }) {
  if (!result.is_cassava) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="rounded-2xl border-2 border-red-200 bg-red-50 p-8 flex flex-col items-center gap-4 text-center shadow-lg shadow-red-500/5"
      >
        <motion.div animate={{ rotate: [0, -10, 10, 0] }} transition={{ duration: 0.5 }}>
          <XCircle className="w-16 h-16 text-red-400" />
        </motion.div>
        <h2 className="font-bold text-xl text-red-700">Not a cassava leaf</h2>
        <p className="text-red-500 text-sm max-w-sm">{result.error || 'Please upload a clear photo of a cassava leaf.'}</p>
      </motion.div>
    )
  }

  const topKey = result.top5?.[0]?.name || 'healthy'
  const style = DISEASE_STYLE[topKey] || DISEASE_STYLE.healthy
  const isHealthy = topKey === 'healthy'
  const pct = Math.round(result.confidence * 100)

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
      {/* Main card */}
      <div className={`rounded-2xl border-2 ${style.border} ${style.bg} p-5 shadow-sm`}>
        <div className="flex items-center gap-4">
          <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 300 }} className="text-5xl">
            {style.emoji}
          </motion.span>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              {isHealthy ? <CheckCircle className="w-5 h-5 text-blue-500" /> : <AlertTriangle className="w-5 h-5 text-amber-500" />}
              <h2 className={`font-black text-lg ${style.text}`}>{result.disease}</h2>
            </div>
            <div className="flex items-center gap-2 mt-1.5">
              <div className="h-2 w-28 rounded-full bg-white/80 overflow-hidden border border-slate-200">
                <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 1 }} className="h-full rounded-full bg-gradient-to-r from-blue-400 to-indigo-500" />
              </div>
              <span className="text-sm font-bold text-slate-600">{pct}%</span>
            </div>
          </div>
        </div>

        <div className="mt-4 space-y-1.5">
          {result.top5?.map((item, i) => (
            <ConfidenceBar key={item.name} name={item.name} confidence={item.confidence} isTop={i === 0} delay={i} />
          ))}
        </div>
      </div>

      {/* Advice */}
      {result.advice && (
        <Section title="Farmer Advice" icon={Shield} defaultOpen={true}>
          <div className="space-y-3">
            {Object.entries(result.advice).map(([section, text], i) => (
              <motion.div key={section} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}>
                <p className="text-[10px] font-black uppercase tracking-widest text-blue-500 mb-0.5">{section}</p>
                <p className="text-slate-600 text-sm leading-relaxed">{text}</p>
              </motion.div>
            ))}
          </div>
        </Section>
      )}

      {/* Web */}
      {result.web_results?.length > 0 && (
        <Section title={`Web Resources (${result.web_results.length})`} icon={ExternalLink}>
          <div className="space-y-2">
            {result.web_results.map((r, i) => (
              <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                className="block p-3 rounded-xl bg-slate-50 border border-slate-100 hover:border-blue-200 hover:bg-blue-50/50 transition-all group">
                <p className="text-sm font-medium text-slate-700 group-hover:text-blue-600 flex items-center gap-1 transition-colors">
                  {r.title} <ExternalLink className="w-3 h-3 opacity-40" />
                </p>
                <p className="text-xs text-slate-400 mt-0.5 line-clamp-2">{r.snippet}</p>
              </a>
            ))}
          </div>
        </Section>
      )}

      {/* Feedback */}
      <Section title="Rate this diagnosis" icon={CheckCircle}>
        <FeedbackPanel result={result} imageFile={imageFile} />
      </Section>
    </motion.div>
  )
}
