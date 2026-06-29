# Task: Replace 2D Gene Network with Interactive 3D Visualization

## Context

This is a BRCA breast cancer subtype classification dashboard (React + FastAPI).  
The prediction tab currently shows a **2D gene network** (Cytoscape.js) with top-50 attention-weighted
gene-gene interactions. The goal is to replace it with a **state-of-the-art 3D interactive visualization**
using `react-force-graph` (Three.js under the hood), with rich hover tooltips that distinguish between
data coming from the GNN model vs the STRING database.

The visualization shows **per-patient** attention — which gene interactions the GATv2Conv model
focused on when classifying this specific patient's tumour subtype.

---

## Data Sources (Critical for Tooltip Design)

Two completely different systems contribute data:

| Data | Source | Field | Meaning |
|---|---|---|---|
| `weight` (on each edge) | **GNN Model** — GATv2Conv Layer 3 attention | `top_edges[i].weight` | How much the model focused on this gene-gene interaction for THIS patient |
| `string_score` (on each edge) | **STRING DB** — biological PPI confidence | `top_edges[i].string_score` | How well-established this protein-protein interaction is in biology (0.4=low, 1.0=very high) |

These must be visually distinguished in tooltips: model data = dynamic, patient-specific; STRING data = static, biological ground truth.

---

## Files to Modify / Create

| File | Action |
|---|---|
| `ui/backend/main.py` | Add `ppi_score_lookup` + `ppi_degree` + `string_score` field to `top_attention_edges()` |
| `ui/frontend/src/components/NetworkGraph3D.jsx` | **CREATE** — new 3D component (replaces NetworkGraph.jsx) |
| `ui/frontend/src/components/PredictionTab.jsx` | Change import + swap component |
| `ui/frontend/package.json` + `node_modules` | Install `react-force-graph` |

Do NOT delete or modify `NetworkGraph.jsx` (keep it for reference / rollback).

---

## 1. Backend Change — `ui/backend/main.py`

### 1a. Add lookup structures in `load_resources()`

After the `edge_index` / `edge_attr` block in `load_resources()` (after the for-loop that builds edges),
add these two structures:

```python
# STRING score lookup for tooltip enrichment
global ppi_score_lookup, ppi_degree
ppi_score_lookup = {}
for _, row in ppi_df.iterrows():
    g1, g2 = str(row['node1']), str(row['node2'])
    sc = float(row['combined_score'])
    ppi_score_lookup[(g1, g2)] = sc
    ppi_score_lookup[(g2, g1)] = sc

# Per-gene PPI degree (number of STRING neighbours)
ppi_degree = {}
for g in gene_list:
    ppi_degree[g] = int(
        ((ppi_df['node1'] == g) | (ppi_df['node2'] == g)).sum()
    )
```

Also declare them as globals at the top of the file alongside the other globals:
```python
ppi_score_lookup = None
ppi_degree       = None
```

### 1b. Enrich `top_attention_edges()` return value

Modify the function to also return `string_score` and per-node `ppi_degree`.  
Replace the current `results.append({...})` inside `top_attention_edges()` with:

```python
g_src = gene_list[s]
g_dst = gene_list[d]
str_sc = ppi_score_lookup.get((g_src, g_dst))  # may be None if not in STRING
results.append({
    "source":       g_src,
    "target":       g_dst,
    "weight":       round(sc, 6),
    "string_score": round(str_sc, 3) if str_sc is not None else None,
    "src_degree":   ppi_degree.get(g_src, 0),
    "dst_degree":   ppi_degree.get(g_dst, 0),
})
```

No changes needed to the `/predict` endpoint itself — it already calls `top_attention_edges(attn)`
and passes the result through.

---

## 2. Install npm Dependency

```bash
cd ui/frontend
npm install react-force-graph
```

`react-force-graph` wraps Three.js and provides `ForceGraph3D`. It has no conflicting peer deps
with the existing stack (React 18, Vite 5).

---

## 3. Create `NetworkGraph3D.jsx`

Create `ui/frontend/src/components/NetworkGraph3D.jsx` with the full implementation below.
Read it carefully — every prop and callback is intentional.

```jsx
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph'

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

function Tooltip({ tooltip }) {
  if (!tooltip) return null
  const { x, y, type, data } = tooltip

  // Clamp to viewport so it doesn't overflow right/bottom
  const style = {
    position: 'fixed',
    left: Math.min(x + 14, window.innerWidth - 280),
    top:  Math.min(y + 14, window.innerHeight - 200),
    zIndex: 9999,
    pointerEvents: 'none',
  }

  return (
    <div style={style}
      className="w-64 rounded-xl border border-white/15 text-xs leading-relaxed shadow-2xl"
      style={{
        ...style,
        background: 'rgba(10, 20, 40, 0.92)',
        backdropFilter: 'blur(12px)',
      }}
    >
      {type === 'node' ? <NodeTooltip data={data} /> : <EdgeTooltip data={data} />}
    </div>
  )
}

function NodeTooltip({ data }) {
  const isCritical = CRITICAL_GENES.has(data.id)
  return (
    <div className="p-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-bold text-white">{data.id}</span>
        {isCritical && (
          <span className="px-1.5 py-0.5 rounded text-amber-300 border border-amber-400/40"
                style={{ fontSize: '10px', background: 'rgba(251,191,36,0.1)' }}>
            ★ критичен
          </span>
        )}
      </div>

      <div className="border-t border-white/10 pt-2 space-y-1.5">
        <p className="text-slate-400 uppercase tracking-wider" style={{ fontSize: '9px' }}>
          от модела — GATv2Conv Layer 3
        </p>
        <div className="flex justify-between">
          <span className="text-slate-300">Внимание (сума)</span>
          <span className="text-cyan-400 font-mono">{data.attentionSum.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-300">Връзки в граф</span>
          <span className="text-cyan-400 font-mono">{data.edgeCount}</span>
        </div>
      </div>

      <div className="border-t border-white/10 pt-2 space-y-1.5">
        <p className="text-slate-400 uppercase tracking-wider" style={{ fontSize: '9px' }}>
          от STRING DB — PPI мрежа
        </p>
        <div className="flex justify-between">
          <span className="text-slate-300">PPI степен</span>
          <span className="text-emerald-400 font-mono">{data.ppiDegree}</span>
        </div>
      </div>
    </div>
  )
}

function EdgeTooltip({ data }) {
  const scoreColor = data.stringScore >= 0.7 ? '#34d399'
                   : data.stringScore >= 0.5 ? '#fbbf24'
                   :                           '#f87171'
  const scoreLabel = data.stringScore >= 0.7 ? 'висока'
                   : data.stringScore >= 0.5 ? 'средна'
                   :                           'ниска'
  return (
    <div className="p-3 space-y-2">
      <p className="text-sm font-semibold text-white">
        {data.source} → {data.target}
      </p>

      <div className="border-t border-white/10 pt-2 space-y-1.5">
        <p className="text-slate-400 uppercase tracking-wider" style={{ fontSize: '9px' }}>
          от модела — GATv2Conv attention
        </p>
        <div className="flex justify-between items-center">
          <span className="text-slate-300">Тежест</span>
          <span className="text-cyan-400 font-mono">{data.weight.toFixed(6)}</span>
        </div>
        <p className="text-slate-500" style={{ fontSize: '10px' }}>
          Колко се е фокусирал моделът върху тази взаимовръзка за този пациент
        </p>
      </div>

      <div className="border-t border-white/10 pt-2 space-y-1.5">
        <p className="text-slate-400 uppercase tracking-wider" style={{ fontSize: '9px' }}>
          от STRING DB v12 — биологична достоверност
        </p>
        {data.stringScore != null ? (
          <>
            <div className="flex justify-between items-center">
              <span className="text-slate-300">combined_score</span>
              <span className="font-mono font-bold" style={{ color: scoreColor }}>
                {data.stringScore.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-300">Увереност</span>
              <span className="font-semibold" style={{ color: scoreColor, fontSize: '11px' }}>
                {scoreLabel}
              </span>
            </div>
            <p className="text-slate-500" style={{ fontSize: '10px' }}>
              Независима от модела — колко добре е доказано взаимодействието в литературата
            </p>
          </>
        ) : (
          <p className="text-slate-500 italic" style={{ fontSize: '10px' }}>Не е в STRING top-50</p>
        )}
      </div>
    </div>
  )
}


export default function NetworkGraph3D({ edges, prediction }) {
  const fgRef   = useRef()
  const [tooltip, setTooltip] = useState(null)
  const mousePos = useRef({ x: 0, y: 0 })

  const color = SUBTYPE_PALETTE[prediction] ?? '#0ea5e9'

  // Track mouse globally for tooltip positioning
  useEffect(() => {
    const move = e => { mousePos.current = { x: e.clientX, y: e.clientY } }
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [])

  // Build graph data with derived node statistics
  const graphData = useMemo(() => {
    if (!edges?.length) return { nodes: [], links: [] }

    const maxW = Math.max(...edges.map(e => e.weight))
    const minW = Math.min(...edges.map(e => e.weight))
    const norm = v => maxW === minW ? 0.5 : (v - minW) / (maxW - minW)

    // Aggregate per-node stats from edges
    const nodeStats = {}
    edges.forEach(e => {
      for (const [gene, role] of [[e.source, 'src'], [e.target, 'dst']]) {
        if (!nodeStats[gene]) nodeStats[gene] = { attentionSum: 0, edgeCount: 0, ppiDegree: 0 }
        nodeStats[gene].attentionSum += e.weight
        nodeStats[gene].edgeCount   += 1
        // Use degree from first edge that provides it (src_degree / dst_degree)
        if (role === 'src' && e.src_degree) nodeStats[gene].ppiDegree = e.src_degree
        if (role === 'dst' && e.dst_degree) nodeStats[gene].ppiDegree = e.dst_degree
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

  // After graph stabilises, zoom to fit
  useEffect(() => {
    if (!fgRef.current) return
    const t = setTimeout(() => fgRef.current?.zoomToFit(600, 80), 1200)
    return () => clearTimeout(t)
  }, [graphData])

  const handleNodeHover = useCallback(node => {
    if (!node) { setTooltip(null); return }
    setTooltip({
      type: 'node',
      x: mousePos.current.x,
      y: mousePos.current.y,
      data: node,
    })
  }, [])

  const handleLinkHover = useCallback(link => {
    if (!link) { setTooltip(null); return }
    setTooltip({
      type: 'edge',
      x: mousePos.current.x,
      y: mousePos.current.y,
      data: link,
    })
  }, [])

  // Continuously update tooltip position while hovering
  useEffect(() => {
    if (!tooltip) return
    const move = e => setTooltip(prev => prev ? { ...prev, x: e.clientX, y: e.clientY } : null)
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [!!tooltip])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        backgroundColor="#0a1628"

        // Nodes
        nodeLabel={() => ''}                         // we use custom tooltip
        nodeColor={node => node.isCritical ? '#fbbf24' : color}
        nodeVal={node => 3 + node.attentionSum * 40} // size ∝ attention
        nodeOpacity={0.92}
        nodeResolution={16}

        // Edges
        linkColor={link => `${color}aa`}
        linkWidth={link => 0.5 + link.norm * 4}
        linkOpacity={0.55}

        // Animated particles flowing along edges — looks impressive
        linkDirectionalParticles={link => Math.round(1 + link.norm * 3)}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleWidth={link => 1 + link.norm * 2}
        linkDirectionalParticleColor={() => color}

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

      {/* Tooltip portal — rendered outside ForceGraph3D canvas */}
      <Tooltip tooltip={tooltip} />

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
```

---

## 4. Update `PredictionTab.jsx`

Two changes only:

### 4a. Replace import at top of file

```jsx
// OLD:
import NetworkGraph from './NetworkGraph'

// NEW:
import NetworkGraph3D from './NetworkGraph3D'
```

### 4b. Replace component usage (~line 184)

```jsx
// OLD:
<NetworkGraph edges={current.top_edges} prediction={current.prediction} />

// NEW:
<NetworkGraph3D edges={current.top_edges} prediction={current.prediction} />
```

No other changes to `PredictionTab.jsx`.

---

## 5. Tooltip Styling Note

The `Tooltip` component uses both Tailwind classes AND inline `style={{}}` for the backdrop-filter
(since Tailwind's `backdrop-blur` requires the `@tailwindcss/backdrop-filter` plugin which may not
be configured). The inline style is intentional — do not replace it with Tailwind classes.

The `Tooltip` component uses `position: fixed` so it renders relative to the viewport, not the
graph container. This ensures it's never clipped by the `overflow: hidden` of the parent panel.

---

## 6. Verification

### Step 1 — Restart backend
```bash
# In project root
uvicorn ui.backend.main:app --reload --port 8000
```

### Step 2 — Verify new fields in API response
```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@sample_patients_prediction.csv" | python -m json.tool | python -c "
import json, sys
data = json.load(sys.stdin)
edge = data[0]['top_edges'][0]
print('Edge fields:', list(edge.keys()))
print('Sample edge:', edge)
"
```
Expected output must include `string_score` and `src_degree` / `dst_degree`:
```
Edge fields: ['source', 'target', 'weight', 'string_score', 'src_degree', 'dst_degree']
Sample edge: {'source': 'FGFR1', 'target': 'ERBB2', 'weight': 0.084231, 'string_score': 0.876, 'src_degree': 12, 'dst_degree': 18}
```

### Step 3 — Visual checks in browser (http://localhost:5173)
1. Upload `sample_patients_prediction.csv`
2. Select the Basal patient (TCGA-A1-A0SK-01)
3. The right panel should render a **3D rotating graph** with glowing nodes
4. Animated particles should flow along edges
5. Hover over a **node** → tooltip with two sections: "ОТ МОДЕЛА" (cyan values) + "ОТ STRING DB" (green values)
6. Hover over an **edge** → tooltip showing both `weight` (model) and `string_score` (STRING) with confidence label (висока/средна/ниска)
7. Critical genes (ERBB2, FOXA1, etc.) should have **amber/gold** colour instead of subtype colour
8. Drag nodes, zoom, orbit — all navigation should work

### Step 4 — Edge case: missing string_score
For edges where STRING has no direct entry (rare), `string_score` will be `null`.
The tooltip should show "Не е в STRING top-50" (italic grey text) — verify this renders correctly.

---

## 7. What NOT to Do

- Do NOT delete `NetworkGraph.jsx`
- Do NOT add TypeScript or type annotations
- Do NOT add additional npm packages (Three.js is bundled inside `react-force-graph`)
- Do NOT change any backend endpoint URLs or response keys other than adding the three new fields
- Do NOT modify `BrainTab.jsx` (it has its own separate 3D visualization for the full model)
- Do NOT add custom Three.js `nodeThreeObject` — the default sphere renderer is sufficient and simpler
- Do NOT add bloom post-processing (`UnrealBloomPass`) — it requires manual Three.js renderer access
  and adds significant complexity; the particle effect alone is visually compelling
