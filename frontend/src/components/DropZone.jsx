import { useDropzone } from 'react-dropzone'
import { Upload, ImagePlus, Check } from 'lucide-react'
import { motion } from 'framer-motion'

export default function DropZone({ onFile, preview }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    onDrop: (files) => files[0] && onFile(files[0]),
  })

  return (
    <motion.div
      whileHover={{ scale: 1.005 }}
      whileTap={{ scale: 0.995 }}
      {...getRootProps()}
      className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden min-h-[220px] flex items-center justify-center bg-white
        ${isDragActive
          ? 'border-blue-400 bg-blue-50 shadow-[0_0_30px_rgba(59,130,246,0.12)]'
          : preview
            ? 'border-blue-200'
            : 'border-slate-200 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-500/5'}`}
    >
      <input {...getInputProps()} />
      {preview ? (
        <div className="relative group w-full">
          <img src={preview} alt="preview" className="w-full max-h-[280px] object-cover" />
          <div className="absolute inset-0 bg-slate-900/50 opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center justify-center backdrop-blur-sm">
            <div className="flex items-center gap-2 bg-white/90 px-4 py-2 rounded-full shadow-lg">
              <ImagePlus className="w-4 h-4 text-blue-600" />
              <span className="text-slate-700 font-semibold text-sm">Change image</span>
            </div>
          </div>
          <div className="absolute top-3 right-3 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center shadow-lg">
            <Check className="w-4 h-4 text-white" />
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4 py-12 px-6">
          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
            className="w-16 h-16 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center"
          >
            <Upload className="w-7 h-7 text-blue-500" />
          </motion.div>
          <div className="text-center">
            <p className="font-bold text-slate-700 text-lg">
              {isDragActive ? 'Drop it here!' : 'Upload cassava leaf'}
            </p>
            <p className="text-slate-400 text-sm mt-1">Drag & drop or tap to browse</p>
          </div>
          <div className="flex gap-2">
            {['JPG', 'PNG', 'WEBP'].map(fmt => (
              <span key={fmt} className="text-[10px] font-bold px-2.5 py-1 rounded-md bg-slate-100 text-slate-500">{fmt}</span>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}
