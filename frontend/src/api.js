import axios from 'axios'
export const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const api = axios.create({baseURL: BASE_URL + '/api', timeout: 50000})
api.interceptors.request.use(config => { const token = localStorage.getItem('ds_token'); if(token) config.headers.Authorization = `Bearer ${token}`; return config })
api.interceptors.response.use(r => r, error => { if(error.response?.status === 401 && !error.config?.url?.includes('/auth/login')){localStorage.removeItem('ds_token'); localStorage.removeItem('ds_user'); window.dispatchEvent(new Event('ds-logout'))} return Promise.reject(error) })
export function errorText(error) {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map(item => `${item.loc?.at(-1) || 'Input'}: ${item.msg}`).join('. ')
  }
  if (!error?.response) return 'The service could not be reached. Please try again.'
  return 'Something went wrong. Please try again.'
}
export default api
