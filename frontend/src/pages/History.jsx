import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Trash2, BarChart2, CheckCircle, AlertTriangle, Clock } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'

const COLORS = ['#3b82f6', '#eab308', '#ef4444', '#f97316', '#14b8a6']
const NAMES = { healthy: 'Healthy', mosaic_disease: 'Mosaic', bacterial_blight: 'Blight', brown_streak_disease: 'Streak', green_mottle: 'Mottle' }

export default function History() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    setHistory(JSON.parse(localStorage.getItem('cassava_history') || '[]'))
  }, [])

  function clear() {
    localStorage.removeItem('cassava_history')
    setHistory([])
  }

  const total = history.length
  const counts = history.reduce((acc, h) => {
    const key = h.top5?.[0]?.name || 'unknown'
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {})
  const chartData = Object.entries(counts).map(([name, value], i) => ({ name: NAMES[name] || name, value, fill: COLORS[i % COLORS.length] }))

  if (!total) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center">
        <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}>
          <div className="w-20 h-20 rounded-2xl border-2 border-dashed border-slate-200 bg-white flex items-center justify-center mx-auto mb-5">
            <BarChart2 className="w-10 h-10 text-slate-200" />
          </div>
        </motion.div>
        <h2 className="text-2xl font-bold text-slate-600 mb-2">No history yet</h2>
        <p className="text-slate-400">Analyze some leaves and they'll appear here.</p>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 md:py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-black text-slate-800">History</h1>
          <p className="text-slate-400 text-sm mt-1">{total} scan{total !== 1 ? 's' : ''}</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={clear}
          className="flex items-center gap-2 px-4 py-2 rounded-xl border border-red-200 text-red-500 text-sm font-medium hover:bg-red-50 transition-all"
        >
          <Trash2 className="w-4 h-4" /> Clear
        </motion.button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {[
          { label: 'Total Scans', value: total, color: 'text-blue-600' },
          { label: 'Healthy', value: counts.healthy || 0, color: 'text-blue-600' },
          { label: 'Diseased', value: total - (counts.healthy || 0), color: 'text-red-500' },
          { label: 'Types Found', value: Object.keys(counts).length, color: 'text-violet-600' },
        ].map(({ label, value, color }) => (
          <motion.div key={label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-xl border border-slate-200 p-4 text-center shadow-sm">
            <p className={`text-3xl font-black ${color}`}>{value}</p>
            <p className="text-[10px] uppercase tracking-wider text-slate-400 mt-1">{label}</p>
          </motion.div>
        ))}
      </div>

      {/* Chart + list */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <p className="text-sm font-bold text-slate-600 mb-3">Distribution</p>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={chartData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" stroke="none">
                {chartData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 mt-3">
            {chartData.map(({ name, fill, value }) => (
              <span key={name} className="flex items-center gap-1.5 text-xs text-slate-500">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: fill }} />
                {name} ({value})
              </span>
            ))}
          </div>
        </div>

        <div className="lg:col-span-2 space-y-2 max-h-[500px] overflow-y-auto pr-1">
          {history.map((item, i) => {
            const key = item.top5?.[0]?.name || 'healthy'
            const isH = key === 'healthy'
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: Math.min(i * 0.03, 0.5) }}
                className="bg-white rounded-xl border border-slate-200 px-4 py-3 flex items-center gap-3 shadow-sm hover:shadow-md hover:border-blue-200 transition-all"
              >
                {isH ? <CheckCircle className="w-5 h-5 text-blue-500 shrink-0" /> : <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0" />}
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-slate-700 text-sm truncate">{item.disease || 'Unknown'}</p>
                  <p className="text-[10px] text-slate-400 flex items-center gap-1"><Clock className="w-3 h-3" />{new Date(item.timestamp).toLocaleString()}</p>
                </div>
                <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${isH ? 'bg-blue-50 text-blue-600 border border-blue-200' : 'bg-amber-50 text-amber-600 border border-amber-200'}`}>
                  {Math.round(item.confidence * 100)}%
                </span>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
