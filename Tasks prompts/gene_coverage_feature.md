# Task: Add Gene Coverage Panel to Prediction Dashboard

## Context

This is a BRCA breast cancer subtype classification project using a Graph Neural Network (GATv2Conv, 3 layers).
The model was trained on **196 genes** from TCGA BRCA Pan-Cancer Atlas 2018 (981 patients).
The prediction dashboard (`/predict` endpoint + React frontend) already works end-to-end.

The goal of this task is to add a **Gene Coverage** panel that tells the user:
1. How many of their submitted genes (out of 196) were recognised by the model
2. Which **critical high-attention genes** are missing from the submission (if any)

This is important because the model silently imputes missing genes as `0.0` (mean expression). The user
currently has no feedback about data quality. Missing critical genes degrade prediction accuracy.

---

## Files to Modify

| File | Change |
|---|---|
| `ui/backend/main.py` | Add `CRITICAL_GENES` constant + coverage fields to `/predict` response |
| `ui/frontend/src/components/PredictionTab.jsx` | Add `GeneCoverage` UI section |

Do NOT modify any other files. Do NOT change training scripts, model architecture, or other endpoints.

---

## 1. Backend Change — `ui/backend/main.py`

### 1a. Add `CRITICAL_GENES` constant

Add this constant directly after the `CLASSES` and `DEVICE` definitions (around line 44), before the
`model = None` block:

```python
CRITICAL_GENES = [
    'ERBB2', 'ERBB3', 'FGFR1', 'FGFR2', 'FGFR4',
    'FOXA1', 'FOXC1', 'ESR1',  'MAPK1', 'STAT3',
    'KDR',   'HGF',   'IGF1R', 'TSC2',  'STK11',
    'TP53',  'PIK3CA','PTEN',  'BRCA1', 'BRCA2',
    'EGFR',  'MYC',   'CCND1', 'CDH1',  'GATA3',
    'AR',    'PGR',   'KIT',   'VEGFA', 'MKI67',
]
```

These are the 30 genes with historically high GATv2 attention weights in this model, selected from
attention analysis + PAM50 biology. They are all present in the model's 196-gene list.

### 1b. Compute coverage in `/predict` endpoint

Inside the `for patient_id in df.columns:` loop in the `/predict` endpoint (around line 169),
**after** `build_patient_graph` is called but **before** `results.append(...)`, add:

```python
# Gene coverage analysis
provided_genes    = set(df.index) & set(gene_list)
genes_provided    = len(provided_genes)
coverage_pct      = round(genes_provided / len(gene_list) * 100, 1)
missing_critical  = [g for g in CRITICAL_GENES if g not in provided_genes]
```

Then extend the `results.append({...})` dict with four new fields:

```python
"genes_provided":       genes_provided,
"genes_total":          len(gene_list),
"coverage_pct":         coverage_pct,
"missing_critical_genes": missing_critical,
```

The final `results.append({...})` block should look like this (complete, for clarity):

```python
results.append({
    "patient_id":             patient_id,
    "prediction":             CLASSES[pred_idx],
    "confidence":             round(float(probs[pred_idx]) * 100, 1),
    "probabilities":          {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)},
    "top_edges":              top_edges,
    "brain_layers":           brain_layers,
    "genes_provided":         genes_provided,
    "genes_total":            len(gene_list),
    "coverage_pct":           coverage_pct,
    "missing_critical_genes": missing_critical,
})
```

---

## 2. Frontend Change — `ui/frontend/src/components/PredictionTab.jsx`

### 2a. Insert `GeneCoverage` component

Add the following **self-contained component** at the top of the file, after the `SUBTYPE_INFO` constant
(after line 20) and before the `export default function PredictionTab(...)`:

```jsx
function GeneCoverage({ provided, total, coveragePct, missingCritical }) {
  const barColor =
    coveragePct >= 80 ? '#22c55e' :   // green
    coveragePct >= 50 ? '#eab308' :   // yellow
                        '#ef4444'     // red

  return (
    <div className="glass rounded-2xl p-5">
      <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Генно покритие</p>

      {/* Coverage bar */}
      <div className="flex items-center gap-3 mb-2">
        <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${coveragePct}%`, background: barColor }}
          />
        </div>
        <span className="text-white font-semibold text-sm whitespace-nowrap">
          {provided}/{total}
        </span>
      </div>
      <p className="text-xs text-slate-400 mb-3">
        {provided} от {total} гена разпознати ({coveragePct}%)
      </p>

      {/* Warning for missing critical genes */}
      {missingCritical.length > 0 && (
        <div className="rounded-xl bg-amber-500/10 border border-amber-500/30 p-3">
          <p className="text-xs text-amber-400 font-medium mb-1">
            ⚠️ Липсват {missingCritical.length} критични гена:
          </p>
          <p className="text-xs text-amber-300/80 leading-relaxed">
            {missingCritical.join(', ')}
          </p>
        </div>
      )}
    </div>
  )
}
```

### 2b. Insert `<GeneCoverage>` in the JSX

In the `return (...)` of `PredictionTab`, inside the `{results && (...)}` block, in the left column
(the `col-span-4 space-y-4` div), insert `<GeneCoverage>` **between** the main prediction card and
the probability bars section.

The left column currently has this structure:

```jsx
<div className="col-span-4 space-y-4">
  {/* Patient selector — conditional */}
  {results.length > 1 && ( ... )}

  {/* Main prediction card */}
  <div className={`glass rounded-2xl p-6 border ...`}>
    ...
  </div>

  {/* ← INSERT HERE */}

  {/* Probability bars */}
  <div className="glass rounded-2xl p-5">
    ...
  </div>
</div>
```

Add exactly this JSX between the closing `</div>` of the prediction card and the opening `<div>` of
probability bars:

```jsx
{/* Gene coverage */}
{current.genes_total && (
  <GeneCoverage
    provided={current.genes_provided}
    total={current.genes_total}
    coveragePct={current.coverage_pct}
    missingCritical={current.missing_critical_genes ?? []}
  />
)}
```

The guard `current.genes_total &&` ensures backward compatibility if an older cached response
without these fields is somehow loaded.

---

## 3. Verification Steps

After implementing, restart the backend (`uvicorn ui.backend.main:app --reload --port 8000`) and
verify the following scenarios:

### Test A — Full 196-gene CSV (the existing sample file)
File: `sample_patients_prediction.csv` (already in project root, 196 genes × 5 patients)

Expected:
- Coverage bar: green, 196/196 (100%)
- Text: "196 от 196 гена разпознати (100.0%)"
- NO amber warning box

Smoke test via curl:
```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@sample_patients_prediction.csv" | python -m json.tool | grep -E "genes_provided|coverage|missing"
```
Expected output:
```json
"genes_provided": 196,
"genes_total": 196,
"coverage_pct": 100.0,
"missing_critical_genes": []
```

### Test B — Partial gene CSV (PAM50 only, ~50 genes)
Create a test file by keeping only PAM50 genes from the sample and verify:
- Coverage bar: red, ~50/196 (~25%)
- Amber warning listing missing ERBB2, FGFR1, FOXA1, etc.

### Test C — Visual check in browser
Open http://localhost:5173, upload `sample_patients_prediction.csv`, and confirm:
- New "ГЕННО ПОКРИТИЕ" section appears between diagnosis card and probability bars
- Bar colour is green (full coverage)
- No warning box visible

---

## 4. Style Notes

- Keep the same glassmorphism pattern as the rest of the UI: `glass rounded-2xl p-5`
- Use `tracking-wider` + `uppercase` for section labels, matching `text-xs text-slate-400`
- Colour logic mirrors confidence bar pattern already in the file
- The amber warning box style mirrors the error box already in the component

---

## 5. What NOT to do

- Do NOT change `build_patient_graph()` — coverage is computed in the endpoint, not in that helper
- Do NOT add new API endpoints
- Do NOT modify `NetworkGraph.jsx`, `BrainTab.jsx`, or any other component
- Do NOT add npm dependencies
- Do NOT change model loading or inference logic
- Do NOT add TypeScript types (project uses plain JSX)
