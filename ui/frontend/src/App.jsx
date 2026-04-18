import React, { useState } from 'react'
import PredictionTab from './components/PredictionTab'
import DiscoveriesTab from './components/DiscoveriesTab'

const TABS = [
  { id: 'predict',     label: '🔬 Предсказване',           desc: 'Анализ на пациент' },
  { id: 'discoveries', label: '🧬 Биологични закономерности', desc: 'Нови открития' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('predict')

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="glass border-b border-white/10 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-clinical-500 flex items-center justify-center text-white font-bold text-lg glow">
            G
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">BRCA Subtype GNN</h1>
            <p className="text-xs text-slate-400">Graph Neural Network · Breast Cancer Classification</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="w-2 h-2 bg-green-400 rounded-full inline-block animate-pulse"></span>
          Модел зареден · 160 гена · Tesla T4
        </div>
      </header>

      {/* Tabs */}
      <nav className="px-8 pt-6 flex gap-2">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-clinical-500 text-white glow'
                : 'glass text-slate-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <span>{tab.label}</span>
            <span className="block text-xs opacity-60">{tab.desc}</span>
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="flex-1 px-8 py-6">
        {activeTab === 'predict'      && <PredictionTab />}
        {activeTab === 'discoveries'  && <DiscoveriesTab />}
      </main>
    </div>
  )
}
