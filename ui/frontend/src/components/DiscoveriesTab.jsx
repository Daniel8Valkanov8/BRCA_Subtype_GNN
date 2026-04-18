import React, { useState, useEffect } from 'react'
import axios from 'axios'
import NetworkGraph from './NetworkGraph'
import StatisticsPanel from './StatisticsPanel'

export default function DiscoveriesTab() {
  const [data,         setData]        = useState(null)
  const [loading,      setLoading]     = useState(true)
  const [error,        setError]       = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const [filter,       setFilter]      = useState(0)

  useEffect(() => {
    axios.get('/api/discoveries', { timeout: 120000 })
      .then(r => {
        console.log('Данни от backend:', r.data)
        setData(r.data)
      })
      .catch(e => {
        console.error('Грешка:', e)
        setError(e.response?.data?.detail || e.message || 'Грешка при свързване с backend')
      })
      .finally(() => setLoading(false))
  }, [])

  const filteredEdges = data?.edges?.filter(e => e.weight >= filter) ?? []
  const maxWeight     = data ? Math.max(...data.edges.map(e => e.weight)) : 1

  return (
    <div className="space-y-4">
      {/* Header info */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Анализирани пациенти', value: '200', icon: '👥' },
          { label: 'Генни взаимодействия', value: data?.edges?.length ?? '—', icon: '🔗' },
          { label: 'Уникални гени',         value: data?.nodes?.length ?? '—', icon: '🧬' },
        ].map(({ label, value, icon }) => (
          <div key={label} className="glass rounded-xl p-4 flex items-center gap-4">
            <span className="text-3xl">{icon}</span>
            <div>
              <p className="text-2xl font-bold text-white">{value}</p>
              <p className="text-xs text-slate-400">{label}</p>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div className="glass rounded-2xl p-6 border border-red-500/50 text-red-400">
          ⚠️ {error}
        </div>
      )}

      {loading && (
        <div className="glass rounded-2xl p-12 flex items-center justify-center gap-4">
          <div className="w-8 h-8 border-4 border-clinical-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-slate-300">Агрегиране на attention weights от 200 пациента...</span>
        </div>
      )}

      {data && (
        <div className="grid grid-cols-12 gap-6">
          {/* Network */}
          <div className="col-span-8 space-y-3">
            {/* Filter */}
            <div className="glass rounded-xl px-5 py-3 flex items-center gap-4">
              <span className="text-xs text-slate-400 whitespace-nowrap">Минимален attention:</span>
              <input
                type="range" min={0} max={maxWeight} step={maxWeight / 100}
                value={filter}
                onChange={e => setFilter(parseFloat(e.target.value))}
                className="flex-1 accent-clinical-500"
              />
              <span className="text-xs text-white font-mono w-16 text-right">{filter.toFixed(4)}</span>
              <span className="text-xs text-slate-400">{filteredEdges.length} връзки</span>
            </div>

            <div className="glass rounded-2xl overflow-hidden" style={{ height: '500px', display: 'flex', flexDirection: 'column' }}>
              <div className="px-5 py-3 border-b border-white/10">
                <p className="text-sm font-medium text-white">Глобална PPI мрежа — Биологични закономерности</p>
                <p className="text-xs text-slate-400">Агрегирани GAT attention weights · Кликни на връзка за статистики →</p>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <NetworkGraph
                  edges={filteredEdges}
                  prediction="BRCA_LumA"
                  onEdgeClick={e => setSelectedEdge(e)}
                />
              </div>
            </div>
          </div>

          {/* Stats panel */}
          <div className="col-span-4">
            <StatisticsPanel
              gene1={selectedEdge?.source}
              gene2={selectedEdge?.target}
            />

            {/* Top interactions table */}
            <div className="glass rounded-2xl p-5 mt-4">
              <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Топ 10 взаимодействия</p>
              <div className="space-y-2">
                {data.edges.slice(0, 10).map((e, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedEdge({ source: e.source, target: e.target, weight: e.weight })}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-all hover:bg-white/10 ${
                      selectedEdge?.source === e.source && selectedEdge?.target === e.target
                        ? 'bg-clinical-500/20 border border-clinical-500/40'
                        : 'border border-transparent'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-slate-300 font-medium">{e.source} ↔ {e.target}</span>
                      <span className="text-clinical-500 font-mono">{e.weight.toFixed(4)}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
