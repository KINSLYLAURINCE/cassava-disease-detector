import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Scan, Zap, Globe, RotateCcw, Sparkles, Leaf } from 'lucide-react'
import toast from 'react-hot-toast'
import DropZone from '../components/DropZone'
import ResultCard from '../components/ResultCard'
import { LeafSpinner } from '../components/SkeletonLoader'
import { analyzeLeaf, classifyLeaf } from '../lib/api'

export default function Detect() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('full')
  const [useSearch, setUseSearch] = useState(true)

  function handleFile(f) {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  function reset() {
    setFile(null)
    setPreview(null)
    setResult(null)
  }

  async function run() {
    if (!file) return toast.error('Please upload an image first')
    setLoading(true)
    try {
      let data
      if (mode === 'fast') {
        data = await classifyLeaf(file)
      } else {
        data = await analyzeLeaf(file, useSearch)
      }
      setResult(data)
      const history = JSON.parse(localStorage.getItem('cassava_history') || '[]')
      history.unshift({ ...data, timestamp: new Date().toISOString(), imageName: file.name })
      localStorage.setItem('cassava_history', JSON.stringify(history.slice(0, 50)))
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Analysis failed — is the API running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 md:py-14">
      {/* Hero */}
      <motion.div initial={{ opacity: 0, y: -30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="text-center mb-10 md:mb-14">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 200, delay: 0.2 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-50 border border-blue-200 text-blue-600 text-xs font-bold uppercase tracking-wider mb-5"
        >
          <Sparkles className="w-3.5 h-3.5" />
          AI-Powered Detection
        </motion.div>

        <h1 className="text-4xl sm:text-5xl md:text-7xl font-black leading-[1.1] mb-4">
          <span className="gradient-text">Cassava Disease</span>
          <br />
          <span className="text-slate-800">Detector</span>
        </h1>

        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="text-slate-400 text-base md:text-lg max-w-lg mx-auto">
          Upload a leaf photo — AI identifies the disease and gives farming advice.
        </motion.p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-10">
        {/* Left */}
        <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} className="space-y-4">
          <DropZone onFile={handleFile} preview={preview} />

          {/* Mode */}
          <div className="bg-white rounded-2xl border border-slate-200 p-4 space-y-3 shadow-sm">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">Analysis Mode</p>
            <div className="flex gap-2">
              {[
                { key: 'full', label: 'Full Analysis', icon: Scan, desc: 'AI + LLM + Web' },
                { key: 'fast', label: 'Fast Classify', icon: Zap, desc: 'Classify only' },
              ].map(({ key, label, icon: Icon, desc }) => (
                <motion.button
                  key={key}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setMode(key)}
                  className={`flex-1 flex flex-col items-center gap-1.5 py-3.5 rounded-xl border-2 text-sm font-semibold transition-all
                    ${mode === key
                      ? 'border-blue-400 bg-blue-50 text-blue-600 shadow-md shadow-blue-500/10'
                      : 'border-slate-200 text-slate-400 hover:border-blue-200'}`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{label}</span>
                  <span className="text-[10px] font-normal text-slate-400">{desc}</span>
                </motion.button>
              ))}
            </div>
            {mode === 'full' && (
              <label className="flex items-center gap-2 text-sm text-slate-500 cursor-pointer pt-1">
                <input type="checkbox" checked={useSearch} onChange={(e) => setUseSearch(e.target.checked)} className="rounded accent-blue-500 w-4 h-4" />
                <Globe className="w-4 h-4 text-blue-400" />
                Include web search results
              </label>
            )}
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={run}
              disabled={!file || loading}
              className="flex-1 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-black text-base
                disabled:opacity-30 disabled:cursor-not-allowed shadow-xl shadow-blue-500/25 flex items-center justify-center gap-2 btn-shine"
            >
              {loading ? (
                <><motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}><Leaf className="w-5 h-5" /></motion.div> Analyzing...</>
              ) : (
                <><Scan className="w-5 h-5" /> Analyze Leaf</>
              )}
            </motion.button>
            {(file || result) && (
              <motion.button
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                whileHover={{ scale: 1.1, rotate: -90 }}
                whileTap={{ scale: 0.9 }}
                onClick={reset}
                className="w-14 h-14 rounded-xl border border-slate-200 bg-white text-slate-400 hover:text-red-500 hover:border-red-200 flex items-center justify-center transition-colors shadow-sm"
              >
                <RotateCcw className="w-5 h-5" />
              </motion.button>
            )}
          </div>
        </motion.div>

        {/* Right */}
        <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
          <AnimatePresence mode="wait">
            {loading && (
              <motion.div key="load" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center justify-center py-24">
                <LeafSpinner />
              </motion.div>
            )}
            {!loading && result && (
              <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <ResultCard result={result} imageFile={file} />
              </motion.div>
            )}
            {!loading && !result && (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center py-24 gap-4">
                <motion.div animate={{ y: [0, -8, 0] }} transition={{ duration: 4, repeat: Infinity }}>
                  <div className="w-20 h-20 rounded-2xl border-2 border-dashed border-slate-200 bg-white flex items-center justify-center">
                    <Scan className="w-10 h-10 text-slate-200" />
                  </div>
                </motion.div>
                <p className="text-slate-300 font-medium text-sm">Results will appear here</p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  )
}
