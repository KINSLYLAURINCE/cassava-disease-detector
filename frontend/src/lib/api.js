import axios from 'axios'

// In dev, Vite proxies /api/* → localhost:8000 (no CORS issues).
// In prod, set VITE_API_BASE_URL to your deployed API URL.
const BASE = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL
  : '/api'

const http = axios.create({ baseURL: BASE })

export async function classifyLeaf(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await http.post('/classify', form)
  return data
}

export async function analyzeLeaf(file, search = true) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await http.post(`/analyze?search=${search}`, form)
  return data
}

export async function adviseLeaf(file, question) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await http.post(`/advise?question=${encodeURIComponent(question)}`, form)
  return data
}

export async function submitFeedback(payload) {
  const { data } = await http.post('/feedback', payload)
  return data
}

export async function getFeedbackStats() {
  const { data } = await http.get('/feedback/stats')
  return data
}

export async function getDiseases() {
  const { data } = await http.get('/diseases')
  return data
}

export async function getHealth() {
  const { data } = await http.get('/health')
  return data
}
