import React, { useState, useEffect, useMemo } from 'react'
import Plot from 'react-plotly.js'
import axios from 'axios'

// ── Константи ──────────────────────────────────────────────────────────────────

const LAYER_Z     = { layer1: 0, layer2: 1.8, layer3: 3.6 }
const LAYER_LABEL = { layer1: 'Слой 1', layer2: 'Слой 2', layer3: 'Слой 3' }
const LAYER_DESC  = {
  layer1: '1-hop · Директни PPI съседи',
  layer2: '2-hop · Разширена молекулярна мрежа',
  layer3: 'Global · Финални представяния',
}
const LAYER_COLOR = { layer1: '#60a5fa', layer2: '#c084fc', layer3: '#fb923c' }

// Внимание: 4 тегловни ленти (топ → дъно)
const EDGE_BANDS = [
  { min: 0.75, color: 'rgba(255,90,0,0.90)',  width: 3.0 },
  { min: 0.50, color: 'rgba(255,210,0,0.65)', width: 1.8 },
  { min: 0.25, color: 'rgba(0,170,255,0.40)', width: 1.0 },
  { min: 0.00, color: 'rgba(30,60,100,0.20)', width: 0.5 },
]

const SUBTYPE_COLOR = {
  BRCA_LumA:   '#3b82f6',
  BRCA_LumB:   '#a855f7',
  BRCA_Basal:  '#ef4444',
  BRCA_Her2:   '#f97316',
  BRCA_Normal: '#22c55e',
}

// ── Строй на Plotly traces ──────────────────────────────────────────────────────

function buildTraces(activeData, genePos, layerFilter) {
  if (!activeData || !genePos) return []

  const layers = layerFilter === 'all'
    ? ['layer1', 'layer2', 'layer3']
    : [layerFilter]

  const traces = []

  for (const lname of layers) {
    const z         = LAYER_Z[lname]
    const layerData = activeData.layers[lname]
    if (!layerData) continue

    const nodeAttn   = layerData.node_attention || {}
    const attnValues = activeData.genes.map(g => nodeAttn[g.id] ?? 0)
    const maxA = Math.max(...attnValues, 1e-9)
    const minA = Math.min(...attnValues)
    const span = Math.max(maxA - minA, 1e-9)

    // ── Nodes ──────────────────────────────────────────────────────────────────
    traces.push({
      type:       'scatter3d',
      mode:       'markers+text',
      name:       LAYER_LABEL[lname],
      showlegend: layers.length > 1,
      legendgroup: lname,
      x: activeData.genes.map(g => g.x),
      y: activeData.genes.map(g => g.y),
      z: activeData.genes.map(() => z),
      text: activeData.genes.map(g => {
        const norm = ((nodeAttn[g.id] ?? 0) - minA) / span
        return norm > 0.72 ? g.id : ''
      }),
      textposition: 'top center',
      textfont: { size: 8, color: 'rgba(255,255,255,0.75)', family: 'Inter, sans-serif' },
      customdata: activeData.genes.map(g => [g.id, (nodeAttn[g.id] ?? 0).toFixed(5), LAYER_LABEL[lname]]),
      hovertemplate:
        '<b>%{customdata[0]}</b><br>' +
        'Attention: %{customdata[1]}<br>' +
        '%{customdata[2]}<extra></extra>',
      marker: {
        size:      attnValues.map(v => 3 + ((v - minA) / span) * 9),
        color:     attnValues,
        colorscale: 'Plasma',
        cmin:      minA,
        cmax:      maxA,
        showscale: lname === 'layer3' || layers.length === 1,
        colorbar: {
          title:     { text: 'Attention', font: { color: '#94a3b8', size: 11 } },
          tickfont:  { color: '#94a3b8', size: 9 },
          x: 1.02, len: 0.5, thickness: 10,
          bgcolor: 'rgba(0,0,0,0)',
          bordercolor: 'rgba(255,255,255,0.1)',
        },
        opacity:   0.92,
        line:      { width: 0 },
        symbol:    'circle',
      },
    })

    // ── Edges (по тегловни ленти) ───────────────────────────────────────────────
    const edges = layerData.edges ?? []
    const maxW  = Math.max(...edges.map(e => e.weight), 1e-9)

    for (const band of EDGE_BANDS) {
      const bandEdges = edges.filter(e => (e.weight / maxW) >= band.min)
      const ex = [], ey = [], ez = []
      for (const edge of bandEdges) {
        const src = genePos[edge.source]
        const dst = genePos[edge.target]
        if (!src || !dst) continue
        ex.push(src.x, dst.x, null)
        ey.push(src.y, dst.y, null)
        ez.push(z, z, null)
      }
      if (!ex.length) continue
      traces.push({
        type:       'scatter3d',
        mode:       'lines',
        x: ex, y: ey, z: ez,
        line:       { color: band.color, width: band.width },
        hoverinfo:  'none',
        showlegend: false,
        legendgroup: lname,
      })
    }
  }

  // ── Inter-layer pillars (само при всички слоеве) ──────────────────────────────
  if (layerFilter === 'all') {
    const ranked = activeData.genes
      .map(g => ({
        gene: g,
        total: ['layer1', 'layer2', 'layer3'].reduce(
          (s, l) => s + (activeData.layers[l]?.node_attention?.[g.id] ?? 0), 0
        ),
      }))
      .sort((a, b) => b.total - a.total)
      .slice(0, 20)

    const px = [], py = [], pz = []
    for (const { gene } of ranked) {
      const pos = genePos[gene.id]
      if (!pos) continue
      px.push(pos.x, pos.x, pos.x, null)
      py.push(pos.y, pos.y, pos.y, null)
      pz.push(LAYER_Z.layer1, LAYER_Z.layer2, LAYER_Z.layer3, null)
    }
    traces.push({
      type:       'scatter3d',
      mode:       'lines',
      name:       'Информационен поток',
      x: px, y: py, z: pz,
      line:       { color: 'rgba(255,215,0,0.28)', width: 1.8 },
      hoverinfo:  'none',
      showlegend: false,
    })
  }

  return traces
}

// ── Plotly layout ──────────────────────────────────────────────────────────────

const BASE_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 0, r: 0, t: 10, b: 0 },
  legend: {
    font:    { color: '#94a3b8', size: 11 },
    bgcolor: 'rgba(0,0,0,0)',
    bordercolor: 'rgba(255,255,255,0.1)',
    borderwidth: 1,
    x: 0.01, y: 0.98,
  },
  scene: {
    bgcolor: '#06101e',
    xaxis: { visible: false, showgrid: false, zeroline: false },
    yaxis: { visible: false, showgrid: false, zeroline: false },
    zaxis: {
      tickmode:      'array',
      tickvals:      [LAYER_Z.layer1, LAYER_Z.layer2, LAYER_Z.layer3],
      ticktext:      ['Слой 1', 'Слой 2', 'Слой 3'],
      tickfont:      { color: '#64748b', size: 11, family: 'Inter, sans-serif' },
      gridcolor:     'rgba(30,60,100,0.6)',
      gridwidth:     1,
      showgrid:      true,
      zeroline:      false,
      showticklabels: true,
      title:         { text: '' },
    },
    camera: { eye: { x: 1.55, y: 1.55, z: 0.85 } },
    aspectmode: 'auto',
    annotations: [
      {
        x: -1.1, y: -1.1, z: LAYER_Z.layer1,
        text: '<b>Слой 1</b><br><span style="font-size:9px">Директни съседи</span>',
        showarrow: false,
        font: { color: '#60a5fa', size: 11 },
      },
      {
        x: -1.1, y: -1.1, z: LAYER_Z.layer2,
        text: '<b>Слой 2</b><br><span style="font-size:9px">2-hop мрежа</span>',
        showarrow: false,
        font: { color: '#c084fc', size: 11 },
      },
      {
        x: -1.1, y: -1.1, z: LAYER_Z.layer3,
        text: '<b>Слой 3</b><br><span style="font-size:9px">Глобален контекст</span>',
        showarrow: false,
        font: { color: '#fb923c', size: 11 },
      },
    ],
  },
}

// ── Главен компонент ───────────────────────────────────────────────────────────

export default function BrainTab({ patientBrainData }) {
  const [globalData,   setGlobalData]   = useState(null)
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState(null)
  const [viewMode,     setViewMode]     = useState('global')   // 'global' | 'patient'
  const [layerFilter,  setLayerFilter]  = useState('all')

  // Зареждане на глобалните данни при първо отваряне на таба
  useEffect(() => {
    if (globalData || loading) return
    setLoading(true)
    axios.get('/api/brain', { timeout: 600_000 })
      .then(r => { setGlobalData(r.data); setLoading(false) })
      .catch(e => { setError(e.message || 'Грешка при зареждане'); setLoading(false) })
  }, [])

  // Позиционна карта: gene id → { x, y }
  const genePos = useMemo(() => {
    if (!globalData) return {}
    return Object.fromEntries(globalData.genes.map(g => [g.id, { x: g.x, y: g.y }]))
  }, [globalData])

  // Активни данни: глобални или пациентски (позиции винаги от globalData)
  const activeData = useMemo(() => {
    if (viewMode === 'patient' && patientBrainData && globalData) {
      return { genes: globalData.genes, layers: patientBrainData.brain_layers }
    }
    return globalData
  }, [viewMode, patientBrainData, globalData])

  // Plotly traces
  const plotData = useMemo(
    () => buildTraces(activeData, genePos, layerFilter),
    [activeData, genePos, layerFilter]
  )

  // Топ hub гени (Слой 3)
  const hubGenes = useMemo(() => {
    if (!activeData) return []
    const attn = activeData.layers?.layer3?.node_attention ?? {}
    return Object.entries(attn)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 12)
      .map(([gene, val]) => ({ gene, val }))
  }, [activeData])

  const hasPatient = Boolean(patientBrainData)

  return (
    <div className="space-y-4">

      {/* ── Заглавие + контроли ─────────────────────────────────────────────── */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">Мозъкът на GNN модела — 3D Attention Flow</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            {globalData
              ? `${globalData.genes.length} гена · 3 слоя message passing · ${
                  viewMode === 'global' ? 'усреднено по всички пациенти' : patientBrainData.patient_id
                }`
              : loading ? 'Изчисляване на attention weights…' : ''}
          </p>
        </div>

        <div className="flex gap-2 flex-wrap">
          {/* View mode toggle */}
          <div className="glass rounded-xl flex overflow-hidden border border-white/10">
            <button
              onClick={() => setViewMode('global')}
              className={`px-4 py-2 text-sm transition-all ${
                viewMode === 'global'
                  ? 'bg-clinical-500 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              🌐 Глобален
            </button>
            <button
              onClick={() => hasPatient && setViewMode('patient')}
              title={hasPatient ? '' : 'Анализирайте пациент в Таб 1 първо'}
              className={`px-4 py-2 text-sm transition-all ${
                viewMode === 'patient'
                  ? 'bg-purple-600 text-white'
                  : hasPatient
                    ? 'text-slate-400 hover:text-white'
                    : 'text-slate-700 cursor-not-allowed'
              }`}
            >
              👤 {hasPatient ? patientBrainData.patient_id.slice(0, 12) + '…' : 'Пациент'}
            </button>
          </div>

          {/* Layer filter */}
          <div className="glass rounded-xl flex overflow-hidden border border-white/10">
            {['all', 'layer1', 'layer2', 'layer3'].map(l => (
              <button
                key={l}
                onClick={() => setLayerFilter(l)}
                className={`px-3 py-2 text-xs transition-all border-r border-white/10 last:border-0 ${
                  layerFilter === l ? 'bg-white/20 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {l === 'all' ? 'Всички' : `С${l.slice(-1)}`}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Основно съдържание ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-12 gap-4">

        {/* 3-D Plot */}
        <div
          className="col-span-9 glass rounded-2xl overflow-hidden"
          style={{ height: '640px' }}
        >
          {loading && (
            <div className="h-full flex flex-col items-center justify-center gap-5">
              <div className="relative">
                <div className="w-16 h-16 border-4 border-clinical-500/30 rounded-full" />
                <div className="w-16 h-16 border-4 border-clinical-500 border-t-transparent rounded-full animate-spin absolute inset-0" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-white font-medium">Изчисляване на 3D attention flow…</p>
                <p className="text-slate-500 text-xs">Обработка на всички пациенти · може да отнеме 1–2 мин.</p>
                <p className="text-slate-600 text-xs">Резултатите се кешират — следващото зареждане е мигновено</p>
              </div>
            </div>
          )}

          {error && (
            <div className="h-full flex items-center justify-center text-red-400 gap-2">
              <span>⚠️</span> {error}
            </div>
          )}

          {!loading && !error && plotData.length > 0 && (
            <Plot
              data={plotData}
              layout={BASE_LAYOUT}
              config={{
                responsive:               true,
                displayModeBar:           true,
                displaylogo:              false,
                modeBarButtonsToRemove:   ['resetCameraLastSave3d', 'hoverClosest3d'],
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
            />
          )}

          {!loading && !error && !plotData.length && (
            <div className="h-full flex items-center justify-center text-slate-500">
              Данните се зареждат…
            </div>
          )}
        </div>

        {/* Страничен панел */}
        <div className="col-span-3 space-y-4">

          {/* Легенда за слоевете */}
          <div className="glass rounded-2xl p-4">
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Как работи моделът</p>
            <div className="space-y-4">
              {(['layer1', 'layer2', 'layer3']).map(l => (
                <div key={l} className="flex gap-3">
                  <div
                    className="w-2.5 h-2.5 rounded-full mt-1 flex-shrink-0"
                    style={{ background: LAYER_COLOR[l] }}
                  />
                  <div>
                    <p className="text-xs font-medium" style={{ color: LAYER_COLOR[l] }}>
                      {LAYER_LABEL[l]}
                    </p>
                    <p className="text-xs text-slate-500 leading-relaxed">{LAYER_DESC[l]}</p>
                  </div>
                </div>
              ))}
              <div className="flex gap-3">
                <div className="w-2.5 h-2.5 rounded-full mt-1 flex-shrink-0 bg-yellow-400" />
                <div>
                  <p className="text-xs font-medium text-yellow-400">Жълти стълбове</p>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Топ 20 hub гени — предават сигнал и в трите слоя
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Цветова скала на ребрата */}
          <div className="glass rounded-2xl p-4">
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Attention на ребрата</p>
            <div className="space-y-2">
              {[
                { color: 'rgba(255,90,0,0.9)',   label: 'Много висока' },
                { color: 'rgba(255,210,0,0.75)', label: 'Висока' },
                { color: 'rgba(0,170,255,0.6)',  label: 'Средна' },
                { color: 'rgba(30,60,100,0.5)',  label: 'Ниска' },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2">
                  <div className="h-1.5 w-12 rounded-full" style={{ background: color }} />
                  <span className="text-xs text-slate-400">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Топ hub гени */}
          {hubGenes.length > 0 && (
            <div className="glass rounded-2xl p-4">
              <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">
                Топ hub гени · Слой 3
              </p>
              <div className="space-y-2">
                {hubGenes.map(({ gene, val }, i) => {
                  const maxVal = hubGenes[0]?.val || 1
                  const hue = Math.round(280 - i * 14)
                  return (
                    <div key={gene}>
                      <div className="flex justify-between text-xs mb-0.5">
                        <span className="text-slate-200 font-medium">{gene}</span>
                        <span className="text-slate-500 tabular-nums">{val.toFixed(4)}</span>
                      </div>
                      <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width:      `${(val / maxVal) * 100}%`,
                            background: `hsl(${hue},75%,58%)`,
                          }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Пациентски badge (при patient mode) */}
          {viewMode === 'patient' && patientBrainData && (
            <div className="glass rounded-2xl p-4 border border-purple-500/30">
              <p className="text-xs text-purple-400 uppercase tracking-wider mb-1">Текущ пациент</p>
              <p className="text-white font-medium text-sm truncate">{patientBrainData.patient_id}</p>
              <p
                className="text-sm font-semibold mt-1"
                style={{ color: SUBTYPE_COLOR[patientBrainData.prediction] ?? '#94a3b8' }}
              >
                {patientBrainData.prediction}
              </p>
              <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                Визуализирани са attention weights специфични за този пациент
              </p>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
