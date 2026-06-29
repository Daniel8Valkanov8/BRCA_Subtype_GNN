import React, { useEffect, useLayoutEffect, useRef, useState, useCallback, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'

const SUBTYPE_PALETTE = {
  BRCA_LumA:   '#3b82f6',
  BRCA_LumB:   '#a855f7',
  BRCA_Basal:  '#ef4444',
  BRCA_Her2:   '#f97316',
  BRCA_Normal: '#22c55e',
}

// Genes with historically high attention across the TCGA cohort
const CRITICAL_GENES = new Set([
  'ERBB2','ERBB3','FGFR1','FGFR2','FGFR4','FOXA1','FOXC1','ESR1','MAPK1','STAT3',
  'KDR','HGF','IGF1R','TSC2','STK11','TP53','PIK3CA','PTEN','BRCA1','BRCA2',
  'EGFR','MYC','CCND1','CDH1','GATA3','AR','PGR','KIT','VEGFA','MKI67',
])

const TT_W = 300   // tooltip width (px)
const TT_H = 240   // approx tooltip height for flip math

function NodeTooltip({ data }) {
  const isCritical = CRITICAL_GENES.has(data.id)
  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-base font-bold text-white">{data.id}</span>
        {isCritical && (
          <span className="px-1.5 py-0.5 rounded text-amber-300 border border-amber-400/40"
                style={{ fontSize: '10px', background: 'rgba(251,191,36,0.1)' }}>
            ★ критичен
          </span>
        )}
      </div>

      <div className="border-t border-white/10 pt-2.5 space-y-2">
        <p className="text-[10px] text-cyan-400/80 uppercase tracking-wider font-semibold">
          от модела — GATv2Conv Layer 3
        </p>
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-300">Внимание (сума)</span>
          <span className="text-cyan-400 font-mono text-sm font-semibold">{data.attentionSum.toFixed(4)}</span>
        </div>
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-300">Връзки в граф</span>
          <span className="text-cyan-400 font-mono text-sm font-semibold">{data.edgeCount}</span>
        </div>
      </div>

      <div className="border-t border-white/10 pt-2.5 space-y-2">
        <p className="text-[10px] text-emerald-400/80 uppercase tracking-wider font-semibold">
          от STRING DB — PPI мрежа
        </p>
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-300">PPI степен (съседи)</span>
          <span className="text-emerald-400 font-mono text-sm font-semibold">{data.ppiDegree}</span>
        </div>
      </div>
    </div>
  )
}

function EdgeTooltip({ data }) {
  const hasScore   = data.stringScore != null
  const scoreColor = !hasScore               ? '#94a3b8'
                   : data.stringScore >= 0.7 ? '#34d399'
                   : data.stringScore >= 0.5 ? '#fbbf24'
                   :                           '#f87171'
  const scoreLabel = data.stringScore >= 0.7 ? 'висока'
                   : data.stringScore >= 0.5 ? 'средна'
                   :                           'ниска'
  return (
    <div className="p-4 space-y-3">
      <p className="text-base font-semibold text-white">
        {data.sourceId} <span className="text-slate-500">→</span> {data.targetId}
      </p>

      <div className="border-t border-white/10 pt-2.5 space-y-2">
        <p className="text-[10px] text-cyan-400/80 uppercase tracking-wider font-semibold">
          от модела — GATv2Conv attention
        </p>
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-300">Тежест (attention)</span>
          <span className="text-cyan-400 font-mono text-sm font-semibold">{data.weight.toFixed(6)}</span>
        </div>
        <p className="text-slate-500 text-[11px] leading-snug">
          Колко се е фокусирал моделът върху тази взаимовръзка за този пациент
        </p>
      </div>

      <div className="border-t border-white/10 pt-2.5 space-y-2">
        <p className="text-[10px] text-emerald-400/80 uppercase tracking-wider font-semibold">
          от STRING DB v12 — биологична достоверност
        </p>
        {hasScore ? (
          <>
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300">combined_score</span>
              <span className="font-mono text-sm font-bold" style={{ color: scoreColor }}>
                {data.stringScore.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300">Увереност</span>
              <span className="font-semibold text-sm" style={{ color: scoreColor }}>
                {scoreLabel}
              </span>
            </div>
            <p className="text-slate-500 text-[11px] leading-snug">
              Независима от модела — колко добре е доказано взаимодействието в литературата
            </p>
          </>
        ) : (
          <p className="text-slate-500 italic text-[11px]">Не е директно в STRING</p>
        )}
      </div>
    </div>
  )
}


export default function NetworkGraph3D({ edges, prediction }) {
  const fgRef        = useRef()
  const containerRef = useRef(null)
  const tooltipRef   = useRef(null)
  const mousePos     = useRef({ x: 0, y: 0 })
  const [hovered, setHovered] = useState(null)   // { type, data } | null
  const [dims, setDims]       = useState({ w: 800, h: 480 })

  const color = SUBTYPE_PALETTE[prediction] ?? '#0ea5e9'

  // Size the canvas to its container (ForceGraph3D otherwise uses window size)
  useEffect(() => {
    if (!containerRef.current) return
    const el = containerRef.current
    const update = () => setDims({ w: el.clientWidth || 800, h: el.clientHeight || 480 })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Position the tooltip next to the cursor via direct DOM writes (no re-render),
  // flipping to the opposite side near a viewport edge so it is never clipped.
  const positionTooltip = useCallback(() => {
    const el = tooltipRef.current
    if (!el) return
    const { x, y } = mousePos.current
    const w = el.offsetWidth  || TT_W
    const h = el.offsetHeight || TT_H
    const margin = 16
    let left = x + 18
    let top  = y + 18
    if (left + w > window.innerWidth  - margin) left = Math.max(margin, x - w - 18)
    if (top  + h > window.innerHeight - margin) top  = Math.max(margin, y - h - 18)
    el.style.left = `${left}px`
    el.style.top  = `${top}px`
  }, [])

  // Single mouse tracker: updates cursor ref and repositions the tooltip directly
  useEffect(() => {
    const move = e => {
      mousePos.current = { x: e.clientX, y: e.clientY }
      positionTooltip()
    }
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [positionTooltip])

  // Place the tooltip immediately when it appears (before first paint)
  useLayoutEffect(() => {
    if (hovered) positionTooltip()
  }, [hovered, positionTooltip])

  // Build graph data with derived per-node statistics
  const graphData = useMemo(() => {
    if (!edges?.length) return { nodes: [], links: [] }

    const weights = edges.map(e => e.weight)
    const maxW = Math.max(...weights)
    const minW = Math.min(...weights)
    const norm = v => maxW === minW ? 0.5 : (v - minW) / (maxW - minW)

    // Aggregate per-node stats from the edge list
    const nodeStats = {}
    edges.forEach(e => {
      for (const [gene, deg] of [[e.source, e.src_degree], [e.target, e.dst_degree]]) {
        if (!nodeStats[gene]) nodeStats[gene] = { attentionSum: 0, edgeCount: 0, ppiDegree: 0 }
        nodeStats[gene].attentionSum += e.weight
        nodeStats[gene].edgeCount   += 1
        if (deg != null) nodeStats[gene].ppiDegree = deg
      }
    })

    const nodes = Object.entries(nodeStats).map(([id, stats]) => ({
      id,
      ...stats,
      isCritical: CRITICAL_GENES.has(id),
    }))

    const links = edges.map((e, i) => ({
      id:          `e${i}`,
      source:      e.source,
      target:      e.target,
      weight:      e.weight,
      norm:        norm(e.weight),
      stringScore: e.string_score ?? null,
    }))

    return { nodes, links }
  }, [edges])

  // Fit to view once the layout settles
  useEffect(() => {
    if (!fgRef.current) return
    const t = setTimeout(() => fgRef.current?.zoomToFit(700, 60), 1400)
    return () => clearTimeout(t)
  }, [graphData])

  const handleNodeHover = useCallback(node => {
    setHovered(node ? { type: 'node', data: node } : null)
  }, [])

  const handleLinkHover = useCallback(link => {
    if (!link) { setHovered(null); return }
    // source/target may be resolved to node objects by the force engine
    const sourceId = typeof link.source === 'object' ? link.source.id : link.source
    const targetId = typeof link.target === 'object' ? link.target.id : link.target
    setHovered({ type: 'edge', data: { ...link, sourceId, targetId } })
  }, [])

  // Stable ForceGraph accessors (avoid prop churn → keeps WebGL hover in sync)
  const nodeColorCb     = useCallback(node => node.isCritical ? '#fbbf24' : color, [color])
  const nodeValCb       = useCallback(node => 3 + node.attentionSum * 12, [])
  const linkColorCb     = useCallback(() => `${color}aa`, [color])
  const linkWidthCb     = useCallback(link => 0.5 + link.norm * 4, [])
  const linkParticlesCb = useCallback(link => Math.round(1 + link.norm * 3), [])
  const linkPWidthCb    = useCallback(link => 1 + link.norm * 2, [])
  const linkPColorCb    = useCallback(() => color, [color])
  const emptyLabelCb    = useCallback(() => '', [])

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <ForceGraph3D
        ref={fgRef}
        width={dims.w}
        height={dims.h}
        graphData={graphData}
        backgroundColor="#0a1628"

        // Nodes
        nodeLabel={emptyLabelCb}                       // custom tooltip instead
        nodeColor={nodeColorCb}
        nodeVal={nodeValCb}                            // sphere volume ∝ attention
        nodeOpacity={0.92}
        nodeResolution={16}

        // Edges
        linkColor={linkColorCb}
        linkWidth={linkWidthCb}
        linkOpacity={0.55}

        // Animated particles flowing along edges
        linkDirectionalParticles={linkParticlesCb}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleWidth={linkPWidthCb}
        linkDirectionalParticleColor={linkPColorCb}

        // Hover
        onNodeHover={handleNodeHover}
        onLinkHover={handleLinkHover}

        // Physics
        d3AlphaDecay={0.025}
        d3VelocityDecay={0.35}
        cooldownTicks={120}

        // Controls
        enableNodeDrag={true}
        enableNavigationControls={true}
        showNavInfo={false}
      />

      {/* Tooltip overlay — positioned via direct DOM writes (positionTooltip) */}
      {hovered && (
        <div
          ref={tooltipRef}
          className="rounded-xl border border-white/15 leading-relaxed shadow-2xl"
          style={{
            position: 'fixed',
            left: mousePos.current.x + 18,
            top:  mousePos.current.y + 18,
            width: TT_W,
            zIndex: 9999,
            pointerEvents: 'none',
            background: 'rgba(10, 20, 40, 0.95)',
            backdropFilter: 'blur(14px)',
            WebkitBackdropFilter: 'blur(14px)',
          }}
        >
          {hovered.type === 'node'
            ? <NodeTooltip data={hovered.data} />
            : <EdgeTooltip data={hovered.data} />}
        </div>
      )}

      {/* Legend */}
      <div
        className="absolute bottom-3 left-3 rounded-lg px-3 py-2 space-y-1"
        style={{ background: 'rgba(10,22,40,0.75)', backdropFilter: 'blur(8px)', fontSize: '10px' }}
      >
        <div className="flex items-center gap-2 text-slate-400">
          <span className="w-3 h-3 rounded-full inline-block" style={{ background: color }} />
          <span>Ген (размер = attention)</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <span className="w-3 h-3 rounded-full inline-block bg-amber-400" />
          <span>Критичен ген (★)</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <span className="w-6 h-0.5 inline-block" style={{ background: color }} />
          <span>Връзка (дебелина = attention)</span>
        </div>
      </div>

      {/* Source badge */}
      <div
        className="absolute top-3 right-3 rounded-lg px-2 py-1 text-slate-500"
        style={{ background: 'rgba(10,22,40,0.65)', fontSize: '9px' }}
      >
        🔵 Cyan = модел &nbsp;|&nbsp; 🟢 Зелено = STRING DB
      </div>
    </div>
  )
}
