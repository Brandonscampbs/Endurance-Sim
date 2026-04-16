import { Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import ErrorBoundary from './components/ErrorBoundary'
import LoadingSpinner from './components/LoadingSpinner'
import VerificationPage from './pages/verification/VerificationPage'
import VisualizationPage from './pages/visualization/VisualizationPage'
import SimulatePage from './pages/simulate/SimulatePage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100 flex">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          <ErrorBoundary>
            <Suspense fallback={<LoadingSpinner message="Loading..." />}>
              <Routes>
                <Route path="/" element={<VerificationPage />} />
                <Route path="/visualization" element={<VisualizationPage />} />
                <Route path="/simulate" element={<SimulatePage />} />
              </Routes>
            </Suspense>
          </ErrorBoundary>
        </main>
      </div>
    </BrowserRouter>
  )
}
