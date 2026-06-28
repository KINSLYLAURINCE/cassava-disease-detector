import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ThumbsUp, ThumbsDown, Send, CheckCircle } from 'lucide-react'
import { submitFeedback } from '../lib/api'
import toast from 'react-hot-toast'

const DISEASE_OPTIONS = ['mosaic_disease', 'bacterial_blight', 'brown_streak_disease', 'green_mottle', 'healthy']

export default function FeedbackPanel({ result, imageFile }) {
  const [vote, setVote] = useState(null)
  const [correction, setCorrection] = useState('')
  const [comment, setComment] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)

  async function handleSubmit() {
    if (!vote) return
    setLoading(true)
    try {
      await submitFeedback({
        predicted_disease: result.disease,
        predicted_confidence: result.confidence,
        vote,
        correct_disease: vote === 'wrong' ? correction : null,
        comment,
        image_filename: imageFile?.name || null,
      })
      setSubmitted(true)
      toast.success('Feedback saved!')
    } catch {
      toast.error('Could not save feedback')
    } finally {
      setLoading(false)
    }
  }

  if (submitted) {
    return (
      <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="flex flex-col items-center gap-2 py-6">
        <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 0.5 }}>
          <CheckCircle className="w-12 h-12 text-blue-500" />
        </motion.div>
        <p className="font-bold text-blue-600">Thank you!</p>
        <p className="text-xs text-slate-400">Your feedback helps retrain the model.</p>
      </motion.div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        {[
          { key: 'correct', icon: ThumbsUp, label: 'Correct', active: 'border-blue-400 bg-blue-50 text-blue-600' },
          { key: 'wrong', icon: ThumbsDown, label: 'Wrong', active: 'border-red-400 bg-red-50 text-red-600' },
        ].map(({ key, icon: Icon, label, active }) => (
          <motion.button
            key={key}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVote(key)}
            className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl border-2 text-sm font-semibold transition-all
              ${vote === key ? active : 'border-slate-200 text-slate-400 hover:border-slate-300'}`}
          >
            <Icon className="w-4 h-4" /> {label}
          </motion.button>
        ))}
      </div>

      <AnimatePresence>
        {vote === 'wrong' && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
            <select
              value={correction}
              onChange={(e) => setCorrection(e.target.value)}
              className="w-full bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-700 focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
            >
              <option value="">Correct disease...</option>
              {DISEASE_OPTIONS.map(d => <option key={d} value={d}>{d.replace(/_/g, ' ')}</option>)}
            </select>
          </motion.div>
        )}
      </AnimatePresence>

      <textarea
        rows={2}
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="Optional comment..."
        className="w-full bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-700 resize-none focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 placeholder:text-slate-300"
      />

      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleSubmit}
        disabled={!vote || loading || (vote === 'wrong' && !correction)}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-bold text-sm shadow-lg shadow-blue-500/20
          disabled:opacity-30 disabled:cursor-not-allowed transition-all btn-shine"
      >
        <Send className="w-4 h-4" />
        {loading ? 'Sending...' : 'Submit'}
      </motion.button>
    </div>
  )
}
