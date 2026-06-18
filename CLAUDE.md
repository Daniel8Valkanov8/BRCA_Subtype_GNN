# CLAUDE.md — BRCA GNN Master Thesis

## Project Overview

Graph Neural Network за класификация на 5 молекулярни подтипа на рак на гърдата (BRCA).
**Данни**: TCGA BRCA Pan-Cancer Atlas 2018, 981 пациента, 196 гена, STRING PPI мрежа.
**Задача**: 5-class classification → BRCA_Basal, BRCA_Her2, BRCA_LumA, BRCA_LumB, BRCA_Normal.

---

## Best Model

**`best_model_tcga_196genes_09.06.pt`** — текущ production модел за всякакви изводи, анализи и деплой.

| Checkpoint | Архитектура | Гени | Acc | Macro F1 |
|---|---|---|---|---|
| `best_model_tcga_196genes_09.06.pt` | GATv2Conv 3L | 196 | 0.6861 | 0.5429 |
| `best_model_v4_196genes.pt` | BioGNNv4 (JK + dual readout) | 196 | 0.6932 | 0.6646 |
| `best_model_v5_196genes.pt` | BioGNNv4 + DropEdge | 196 | ~0.69 | ~0.66 |
| `best_model_tcga_160genes.pt` | GATConv 3L (стар) | 160 | 0.5851 | 0.4353 |

---

## Architecture — BioGNNv4 (Current Best)

**Input graph** (per patient):
- **Nodes**: 196 гена
- **Node features (dim=3)**: `[z_score, degree_normalized, mean_edge_weight]`
- **Edges**: 2×2091 directed edges (bidirectional STRING PPI)
- **Edge attributes**: `combined_score` ∈ [0.4, 1.0]

```
Layer 1: GATv2Conv(3 → 128, heads=4, concat=True, edge_dim=1, dropout=0.2) → 512d
          BatchNorm1d(512) → ReLU → h1

Layer 2: GATv2Conv(512 → 128, heads=4, concat=True, edge_dim=1, dropout=0.2) → 512d
          BatchNorm1d(512) → ReLU → h2

Layer 3: GATv2Conv(512 → 128, heads=1, concat=False, edge_dim=1, dropout=0.2) → 128d
          BatchNorm1d(128) → ReLU → h3

JumpingKnowledge: concat([h1, h2, h3]) → 1152d

Global Readout:
  global_mean_pool → 1152d
  global_max_pool  → 1152d
  concat([mean, max]) → 2304d

Classifier:
  Dropout(0.3) → Linear(2304→128) → ReLU → Dropout(0.15) → Linear(128→5)
```

---

## Training Configuration

| Параметър | Стойност |
|---|---|
| Optimizer | AdamW (lr=0.0005, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=60, T_mult=2) |
| Loss | FocalLoss(γ=2.0, α=class_weights) |
| Epochs | 300 (early stopping patience=40) |
| Batch size | 16 |
| CV | 5-Fold StratifiedKFold (seed=42) |
| Oversampling | SMOTE (k=5, само в train fold) |
| Gradient clip | max_norm=1.0 |
| Device | CUDA / CPU auto-detect |

**v5 разлика**: FocalLoss(γ=1.0) + DropEdge(p=0.15) по време на training.

---

## Data Pipeline

```
data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv  ← 981 patients, 5 subtypes
data/tcga_expression_198genes.csv                    ← 196 genes × 981 patients (mRNA z-scores)
data/string_ppi_196genes.tsv                         ← 2091 edges, combined_score [0.4-1.0]
```

**Gene evolution**: 27 → 160 → 196 (добавени 36 PAM50 гена за LumA/LumB разграничаване).

**Preprocessing**:
1. Per-patient z-score нормализация (индивидуално за всеки пациент)
2. Граф = пациент; всеки ген = node; STRING edges = edges
3. Node features = [expression_z, ppi_degree_norm, mean_edge_weight]
4. SMOTE само в train fold (no data leakage)

---

## Per-Class Performance (BioGNNv4, 5-Fold CV)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BRCA_Basal | 0.988 | 0.924 | 0.955 | 171 |
| BRCA_Her2 | 0.659 | 0.692 | 0.675 | 78 |
| BRCA_LumA | 0.954 | 0.543 | 0.692 | 499 |
| BRCA_LumB | 0.478 | 0.873 | 0.618 | 197 |
| BRCA_Normal | 0.263 | 0.694 | 0.382 | 36 |
| **Macro Avg** | **0.668** | **0.745** | **0.664** | **981** |

**Проблемни класове**: Normal (само 36 пациента), LumA↔LumB объркване.

---

## Baseline Comparison (196 гена, 981 пациента)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | **0.9072** | **0.7896** |
| Logistic Regression | 0.8960 | 0.8552 |
| MLP | 0.8919 | 0.8392 |
| BioGNNv4 | 0.6932 | 0.6646 |
| Majority Class | 0.5087 | 0.1349 |

**Ключов извод**: Conventional ML методи значително превъзхождат GNN. Вероятна причина: per-patient z-score нормализацията унищожава глобалния сигнал; PPI структурата не добавя достатъчно информация над expression vectors.

---

## Key Files

| Файл | Роля |
|---|---|
| `src/model.py` | BioGNN (оригинален, 3L GATv2) |
| `src/model_v3.py` | BioGNNv3 (2L GATv2 + JK) |
| `src/data_loader.py` | PyG graph builder |
| `train_colab_v4.py` | BioGNNv4 training (текущ best) |
| `train_colab_v5.py` | BioGNNv4 + DropEdge |
| `baseline_comparison.py` | RF/LogReg/MLP 5-Fold CV |
| `ensemble_threshold.py` | v4+v5 ensemble + threshold tuning |
| `run_attention_analysis_v2.py` | Attention weight analysis |
| `ui/backend/main.py` | FastAPI server (predict, discoveries, brain) |
| `v4_results.txt` | Пълни резултати от BioGNNv4 training |

---

## API Endpoints (FastAPI — ui/backend/main.py)

| Endpoint | Method | Описание |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Upload CSV → subtype + attention weights |
| `/discoveries` | GET | Top edges по mean attention (всички 981 пациента) |
| `/brain` | GET | 3D gene layout + per-layer aggregated attention |
| `/statistics/{gene1}/{gene2}` | GET | Pearson correlation + significance |

---

## Known Issues & Limitations

1. **GNN underperforms baselines** — 23% gap vs Random Forest; likely due to per-patient z-score destroying inter-gene signal
2. **LumA/LumB confusion** — хормонално сходни подтипа; recall на LumA само 54%
3. **Normal клас** — само 36 пациента → F1=0.38; нестабилен
4. **SMOTE + Focal Loss** частично помагат, но не решават имбаланса (13.9×)
5. **Attention analysis** — attention ≠ causal importance; Pearson корелация е слаба (0.3-0.5)

---

## Data Sources

- **TCGA BRCA**: cBioPortal — `brca_tcga_pan_can_atlas_2018`
- **PPI Network**: STRING-DB v12 (Human, confidence ≥ 400, combined_score)
- **PAM50 genes**: Standard breast cancer gene panel (36 гена добавени към 160→196)

---

## Tech Stack

```
Python 3.x
PyTorch + PyTorch Geometric (GATv2Conv)
scikit-learn (baselines, CV, metrics)
imbalanced-learn (SMOTE)
pandas / numpy
FastAPI (backend)
React + Vite + Tailwind CSS (frontend)
TensorBoard (runs/)
Google Colab (GPU training)
```

---

## Development Notes

- Тренирането се прави в Google Colab (GPU); локално само CPU
- Model checkpoints се сваля от Colab → local
- `train_colab_v4.py` е основният training script
- 5-Fold CV задължително; не се тренира на целия dataset без CV
- SMOTE само в train fold — критично за честни резултати
- При нов experiment — нов `_results.txt` файл с пълен log
