import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { AnimatePresence, motion } from 'framer-motion'
import Navbar from './components/Navbar'
import MobileNav from './components/MobileNav'
import Detect from './pages/Detect'
import History from './pages/History'
import About from './pages/About'

const pageVariants = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -16 },
}

function AnimatedRoutes() {
  const location = useLocation()
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      >
        <Routes location={location}>
          <Route path="/" element={<Detect />} />
          <Route path="/history" element={<History />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-center"
        toastOptions={{ style: { borderRadius: '12px', background: '#fff', color: '#1e293b', border: '1px solid #e2e8f0', boxShadow: '0 10px 40px rgba(0,0,0,0.08)' } }}
      />
      <div className="min-h-screen relative bg-[#f8fafc]">
        {/* Ambient background */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-[-15%] right-[-10%] w-[600px] h-[600px] rounded-full bg-blue-400/10 blur-[100px] float-slow" />
          <div className="absolute bottom-[-15%] left-[-10%] w-[500px] h-[500px] rounded-full bg-indigo-400/8 blur-[90px] float-medium" />
          <div className="absolute top-[50%] left-[50%] w-[400px] h-[400px] rounded-full bg-sky-300/6 blur-[80px] float-slow" style={{animationDelay: '2s'}} />
        </div>

        {/* Desktop navbar */}
        <div className="hidden md:block relative z-50">
          <Navbar />
        </div>

        <main className="relative z-10">
          <AnimatedRoutes />
        </main>

        {/* Mobile bottom nav */}
        <div className="block md:hidden">
          <MobileNav />
        </div>
      </div>
    </BrowserRouter>
  )
}
