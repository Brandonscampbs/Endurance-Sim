import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import ErrorBoundary from './components/ErrorBoundary'
import ValidationPage from './pages/validation/ValidationPage'
import VisualizationPage from './pages/visualization/VisualizationPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100 flex">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<ValidationPage />} />
              <Route path="/visualization" element={<VisualizationPage />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </BrowserRouter>
  )
}
