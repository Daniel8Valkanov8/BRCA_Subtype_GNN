import React, { useState, useEffect } from 'react'
import axios from 'axios'

function StatRow({ label, value, highlight }) {
  return (
    <div className={`flex justify-between items-center py-2 border-b border-white/5 ${highlight ? 'text-green-400' : 'text-slate-300'}`}>
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-sm font-mono font-medium">{value}</span>
    </div>
  )
}

export default function StatisticsPanel({ gene1, gene2 }) {
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!gene1 || !gene2) return
    setLoading(true)
    axios.get(`/api/statistics/${gene1}/${gene2}`)
      .then(r => setStats(r.data))
      .catch(() => setStats(null))
      .finally(() => setLoading(false))
  }, [gene1, gene2])

  if (!gene1 || !gene2) return (
    <div className="glass rounded-2xl p-6 h-full flex flex-col items-center justify-center text-center">
      <div className="text-4xl mb-3">👆</div>
      <p className="text-slate-400 text-sm">Кликни върху връзка в мрежата за статистики</p>
    </div>
  )

  if (loading) return (
    <div className="glass rounded-2xl p-6 flex items-center justify-center gap-3">
      <div className="w-5 h-5 border-2 border-clinical-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-slate-400 text-sm">Изчисляване...</span>
    </div>
  )

  if (!stats) return (
    <div className="glass rounded-2xl p-6 text-center text-slate-400 text-sm">
      Грешка при зареждане на статистики
    </div>
  )

  const sigColor = stats.significant ? 'text-green-400' : 'text-yellow-400'

  return (
    <div className="glass rounded-2xl p-5 space-y-4">
      <div>
        <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Статистически анализ</p>
        <p className="text-white font-semibold text-sm">{gene1} ↔ {gene2}</p>
      </div>

      <div className={`rounded-xl px-4 py-3 text-center ${stats.significant ? 'bg-green-500/15 border border-green-500/30' : 'bg-yellow-500/15 border border-yellow-500/30'}`}>
        <p className={`text-xs font-medium ${sigColor}`}>
          {stats.significant ? '✓ Статистически значима връзка' : '⚠ Незначима връзка (p > 0.05)'}
        </p>
      </div>

      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Pearson корелация</p>
        <StatRow label="r (коефициент)"  value={stats.pearson_r}  highlight={Math.abs(stats.pearson_r) > 0.5} />
        <StatRow label="p-value"          value={stats.pearson_p}  highlight={stats.pearson_p < 0.05} />
      </div>

      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Spearman корелация</p>
        <StatRow label="ρ (rho)"          value={stats.spearman_r} highlight={Math.abs(stats.spearman_r) > 0.5} />
        <StatRow label="p-value"          value={stats.spearman_p} highlight={stats.spearman_p < 0.05} />
      </div>

      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Описателна статистика</p>
        <StatRow label={`${gene1} mean`}  value={stats.mean_expr1} />
        <StatRow label={`${gene1} std`}   value={stats.std_expr1}  />
        <StatRow label={`${gene2} mean`}  value={stats.mean_expr2} />
        <StatRow label={`${gene2} std`}   value={stats.std_expr2}  />
        <StatRow label="N пациенти"       value={stats.n_samples}  />
      </div>

      <p className="text-xs text-slate-500 text-center">
        Данни от TCGA · {stats.n_samples} пациента
      </p>
    </div>
  )
}
