import { motion } from 'framer-motion'
import { Brain, Leaf, Globe, MessageSquare, RefreshCw, Zap, Camera, Cpu } from 'lucide-react'

const steps = [
  { icon: Camera, color: 'from-blue-400 to-indigo-500', title: 'Upload', desc: 'Farmer takes a photo of a cassava leaf and uploads it.' },
  { icon: Cpu, color: 'from-sky-400 to-blue-500', title: 'EfficientNet-B4', desc: 'Fine-tuned CNN classifies the disease with 83.45% accuracy.' },
  { icon: Brain, color: 'from-violet-400 to-purple-500', title: 'TinyLlama QLoRA', desc: 'Fine-tuned 1.1B LLM generates symptoms, treatment, and prevention advice.' },
  { icon: Globe, color: 'from-cyan-400 to-sky-500', title: 'DuckDuckGo', desc: 'Live web search adds up-to-date scientific resources.' },
  { icon: MessageSquare, color: 'from-amber-400 to-orange-500', title: 'Feedback', desc: 'Farmer rates the result. Wrong answers go into retraining.' },
  { icon: RefreshCw, color: 'from-teal-400 to-emerald-500', title: 'Retrain', desc: 'Feedback loop fine-tunes both models over time.' },
]

const diseases = [
  { name: 'Cassava Mosaic Disease', short: 'CMD', bg: 'bg-amber-50 border-amber-200', text: 'text-amber-700' },
  { name: 'Bacterial Blight', short: 'CBB', bg: 'bg-red-50 border-red-200', text: 'text-red-700' },
  { name: 'Brown Streak Disease', short: 'CBSD', bg: 'bg-orange-50 border-orange-200', text: 'text-orange-700' },
  { name: 'Green Mottle', short: 'CGM', bg: 'bg-teal-50 border-teal-200', text: 'text-teal-700' },
  { name: 'Healthy', short: '---', bg: 'bg-blue-50 border-blue-200', text: 'text-blue-700' },
]

export default function About() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 md:py-14">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-black mb-3">
          <span className="gradient-text">How It Works</span>
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto">
          A two-model MLOps pipeline helping African farmers detect cassava diseases.
        </p>
      </motion.div>

      {/* Pipeline */}
      <div className="relative mb-16">
        <div className="absolute left-7 top-8 bottom-8 w-px bg-gradient-to-b from-blue-300 via-violet-300 to-teal-300" />
        <div className="space-y-5">
          {steps.map(({ icon: Icon, color, title, desc }, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex gap-4 relative group"
            >
              <motion.div
                whileHover={{ scale: 1.15, rotate: 10 }}
                className={`w-14 h-14 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center shrink-0 z-10 shadow-lg`}
              >
                <Icon className="w-6 h-6 text-white" />
              </motion.div>
              <div className="bg-white rounded-xl border border-slate-200 px-5 py-4 flex-1 shadow-sm hover:shadow-md hover:border-blue-200 transition-all">
                <div className="flex items-center gap-2">
                  <p className="font-bold text-slate-700">{title}</p>
                  <span className="text-[10px] font-mono text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">#{i + 1}</span>
                </div>
                <p className="text-slate-400 text-sm mt-1">{desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Diseases */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mb-12">
        <h2 className="text-xl font-bold text-slate-700 mb-4 flex items-center gap-2">
          <Leaf className="w-5 h-5 text-blue-500" /> Detectable Diseases
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {diseases.map(({ name, short, bg, text }, i) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + i * 0.05 }}
              whileHover={{ scale: 1.02, x: 4 }}
              className={`flex items-center justify-between px-4 py-3 rounded-xl border ${bg} cursor-default transition-all`}
            >
              <span className={`font-semibold text-sm ${text}`}>{name}</span>
              <span className="text-xs font-mono text-slate-400">{short}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Accuracy', value: '83.45%' },
          { label: 'Train Loss', value: '0.107' },
          { label: 'Parameters', value: '1.1B' },
          { label: 'Diseases', value: '5' },
        ].map(({ label, value }, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 + i * 0.1 }}
            className="bg-white rounded-xl border border-slate-200 p-4 text-center shadow-sm"
          >
            <p className="text-2xl font-black text-blue-600">{value}</p>
            <p className="text-[10px] uppercase tracking-wider text-slate-400 mt-1">{label}</p>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
