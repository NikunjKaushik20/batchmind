import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import CopilotChat from './components/CopilotChat'
import Dashboard from './pages/Dashboard'
import BatchExplorer from './pages/BatchExplorer'
import CausalOptimizer from './pages/CausalOptimizer'
import GoldenSignature from './pages/GoldenSignature'
import CarbonLedger from './pages/CarbonLedger'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="app-layout">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="main-content">
        <TopBar onMenuClick={() => setSidebarOpen(o => !o)} />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/batch" element={<BatchExplorer />} />
          <Route path="/optimizer" element={<CausalOptimizer />} />
          <Route path="/golden" element={<GoldenSignature />} />
          <Route path="/carbon" element={<CarbonLedger />} />
        </Routes>
      </div>
      {/* CoPilot — available on all pages */}
      <CopilotChat />
    </div>
  )
}
