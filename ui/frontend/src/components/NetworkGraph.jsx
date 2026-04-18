import React, { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'

const SUBTYPE_PALETTE = {
  BRCA_LumA:   '#3b82f6',
  BRCA_LumB:   '#a855f7',
  BRCA_Basal:  '#ef4444',
  BRCA_Her2:   '#f97316',
  BRCA_Normal: '#22c55e',
}

export default function NetworkGraph({ edges, prediction, onEdgeClick }) {
  const containerRef = useRef(null)
  const cyRef        = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !edges?.length) return

    const color   = SUBTYPE_PALETTE[prediction] ?? '#0ea5e9'
    const maxW    = Math.max(...edges.map(e => e.weight))
    const minW    = Math.min(...edges.map(e => e.weight))
    const norm    = v => maxW === minW ? 0.5 : (v - minW) / (maxW - minW)

    const nodeSet = new Set()
    edges.forEach(e => { nodeSet.add(e.source); nodeSet.add(e.target) })

    const elements = [
      ...[...nodeSet].map(id => ({ data: { id } })),
      ...edges.map((e, i) => ({
        data: {
          id:     `e${i}`,
          source: e.source,
          target: e.target,
          weight: e.weight,
          norm:   norm(e.weight),
        }
      }))
    ]

    if (cyRef.current) cyRef.current.destroy()

    cyRef.current = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color':   color,
            'background-opacity': 0.85,
            'border-width':       2,
            'border-color':       '#ffffff30',
            'label':              'data(id)',
            'color':              '#e2e8f0',
            'font-size':          9,
            'text-valign':        'bottom',
            'text-margin-y':      4,
            'width':              20,
            'height':             20,
          }
        },
        {
          selector: 'edge',
          style: {
            'line-color':   color,
            'opacity':      'data(norm)',
            'width':        ele => 1 + ele.data('norm') * 5,
            'curve-style':  'bezier',
          }
        },
        {
          selector: 'node:selected',
          style: {
            'background-color': '#ffffff',
            'border-color':     color,
            'border-width':     3,
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#ffffff',
            'opacity':    1,
          }
        }
      ],
      layout: {
        name:        'cose',
        animate:     true,
        animationDuration: 600,
        nodeRepulsion: 8000,
        idealEdgeLength: 80,
      },
      userZoomingEnabled:    true,
      userPanningEnabled:    true,
      boxSelectionEnabled:   false,
    })

    cyRef.current.on('tap', 'edge', evt => {
      const e = evt.target.data()
      onEdgeClick?.({ source: e.source, target: e.target, weight: e.weight })
    })

    return () => { cyRef.current?.destroy() }
  }, [edges, prediction])

  return <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />
}
