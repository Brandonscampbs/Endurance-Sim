import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SWRConfig } from 'swr'
import App from './App.tsx'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SWRConfig
      value={{
        revalidateOnFocus: false,
        dedupingInterval: 60000,
      }}
    >
      <App />
    </SWRConfig>
  </StrictMode>,
)
