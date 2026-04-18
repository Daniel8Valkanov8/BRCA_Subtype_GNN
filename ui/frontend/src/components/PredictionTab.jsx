import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import NetworkGraph from './NetworkGraph'

const SUBTYPE_COLORS = {
  BRCA_LumA:   { bg: 'bg-blue-500',   text: 'text-blue-400',   bar: '#3b82f6' },
  BRCA_LumB:   { bg: 'bg-purple-500', text: 'text-purple-400', bar: '#a855f7' },
  BRCA_Basal:  { bg: 'bg-red-500',    text: 'text-red-400',    bar: '#ef4444' },
  BRCA_Her2:   { bg: 'bg-orange-500', text: 'text-orange-400', bar: '#f97316' },
  BRCA_Normal: { bg: 'bg-green-500',  text: 'text-green-400',  bar: '#22c55e' },
}

const SUBTYPE_INFO = {
  BRCA_LumA:   'Луминален A — най-честият подтип, добра прогноза, хормон-рецептор позитивен',
  BRCA_LumB:   'Луминален B — хормон-рецептор позитивен, по-агресивен от LumA',
  BRCA_Basal:  'Базален (Triple Negative) — липса на ER/PR/HER2, агресивен, химиотерапия',
  BRCA_Her2:   'HER2 обогатен — HER2 амплификация, таргетна терапия с трастузумаб',
  BRCA_Normal: 'Нормал-подобен — наподобява нормална тъкан, добра прогноза',
}

export default function PredictionTab() {
  const [loading,  setLoading]  = useState(false)
  const [results,  setResults]  = useState(null)
  const [selected, setSelected] = useState(0)
  const [error,    setError]    = useState(null)

  const onDrop = useCallback(async (files) => {
    if (!files.length) return
    setLoading(true)
    setError(null)
    setResults(null)

    const form = new FormData()
    form.append('file', files[0])

    try {
      const res = await axios.post('/api/predict', form)
      setResults(res.data)
      setSelected(0)
    } catch (e) {
      setError(e.response?.data?.detail || 'Грешка при предсказване')
    } finally {
      setLoading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  })

  const current = results?.[selected]

  return (
    <div className="space-y-6">
      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`glass rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 border-2 border-dashed ${
          isDragActive ? 'border-clinical-500 bg-clinical-500/10 glow' : 'border-white/20 hover:border-clinical-500/50'
        }`}
      >
        <input {...getInputProps()} />
        <div className="text-4xl mb-3">📂</div>
        <p className="text-white font-medium text-lg">
          {isDragActive ? 'Пусни файла тук...' : 'Качи CSV файл с генна експресия'}
        </p>
        <p className="text-slate-400 text-sm mt-1">
          Формат: редове = гени, колони = пациенти (z-score стойности)
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div className="glass rounded-2xl p-8 flex items-center justify-center gap-4">
          <div className="w-8 h-8 border-4 border-clinical-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-slate-300">Анализиране на генна експресия...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="glass rounded-2xl p-6 border border-red-500/50 text-red-400">
          ⚠️ {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="grid grid-cols-12 gap-6">
          {/* Left: patient list + prediction */}
          <div className="col-span-4 space-y-4">
            {/* Patient selector */}
            {results.length > 1 && (
              <div className="glass rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Пациенти ({results.length})</p>
                <div className="space-y-1 max-h-36 overflow-y-auto">
                  {results.map((r, i) => (
                    <button
                      key={i}
                      onClick={() => setSelected(i)}
                      className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                        selected === i ? 'bg-clinical-500 text-white' : 'text-slate-400 hover:bg-white/10'
                      }`}
                    >
                      {r.patient_id}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Main prediction card */}
            <div className={`glass rounded-2xl p-6 border ${SUBTYPE_COLORS[current.prediction] ? 'border-white/20' : 'border-white/10'}`}>
              <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Диагноза</p>
              <div className={`text-2xl font-bold ${SUBTYPE_COLORS[current.prediction]?.text ?? 'text-white'} mb-1`}>
                {current.prediction}
              </div>
              <div className="flex items-center gap-2 mb-4">
                <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${current.confidence}%`,
                      background: SUBTYPE_COLORS[current.prediction]?.bar ?? '#0ea5e9'
                    }}
                  />
                </div>
                <span className="text-white font-semibold text-lg">{current.confidence}%</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {SUBTYPE_INFO[current.prediction]}
              </p>
            </div>

            {/* Probability bars */}
            <div className="glass rounded-2xl p-5">
              <p className="text-xs text-slate-400 uppercase tracking-wider mb-4">Вероятности по подтип</p>
              <div className="space-y-3">
                {Object.entries(current.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <div key={cls}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className={SUBTYPE_COLORS[cls]?.text ?? 'text-slate-300'}>{cls.replace('BRCA_', '')}</span>
                        <span className="text-slate-400">{prob}%</span>
                      </div>
                      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${prob}%`, background: SUBTYPE_COLORS[cls]?.bar ?? '#64748b' }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Right: network graph */}
          <div className="col-span-8 glass rounded-2xl overflow-hidden" style={{ height: '560px' }}>
            <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-white">Генна мрежа — Attention Weights</p>
                <p className="text-xs text-slate-400">Размерът и цветът на ребрата показват важността за диагнозата</p>
              </div>
              <span className="text-xs glass px-3 py-1 rounded-full text-slate-400">
                {current.top_edges.length} връзки
              </span>
            </div>
            <NetworkGraph edges={current.top_edges} prediction={current.prediction} />
          </div>
        </div>
      )}
    </div>
  )
}
