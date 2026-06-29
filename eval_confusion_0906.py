"""
Оценка на best_model_tcga_196genes_09.06.pt върху всички 981 пациента.
Репликира точно препроцесинга от backend-а (per-patient z-score).
Извежда per-class метрики + confusion matrix (кой с кой се бърка).
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.dirname(__file__))
from src.model import BioGNN

ROOT       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, 'best_model_tcga_196genes_09.06.pt')
PPI_PATH   = os.path.join(ROOT, 'data', 'string_ppi_196genes.tsv')
EXPR_PATH  = os.path.join(ROOT, 'data', 'tcga_expression_198genes.csv')
CLIN_PATH  = os.path.join(ROOT, 'data', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
CLASSES    = ['BRCA_Basal', 'BRCA_Her2', 'BRCA_LumA', 'BRCA_LumB', 'BRCA_Normal']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Данни ────────────────────────────────────────────────────────────────────
expr = pd.read_csv(EXPR_PATH, index_col=0)          # гени x пациенти
gene_list = list(expr.index)
gene_map  = {g: i for i, g in enumerate(gene_list)}

clin = pd.read_csv(CLIN_PATH, sep='\t', comment='#').dropna(subset=['Subtype'])
clin = clin.set_index('Sample ID')

# Общи пациенти
common = [p for p in expr.columns if p in clin.index]
print(f"Пациенти: expr={expr.shape[1]}, общи с клинични={len(common)}")
labels = clin.loc[common, 'Subtype'].astype(str)
print("Subtype стойности:", sorted(labels.unique()))

# ── PPI граф ─────────────────────────────────────────────────────────────────
ppi = pd.read_csv(PPI_PATH, sep='\t')
edges, escores = [], []
for _, r in ppi.iterrows():
    if r['node1'] in gene_map and r['node2'] in gene_map:
        i, j = gene_map[r['node1']], gene_map[r['node2']]
        s = float(r['combined_score'])
        edges += [[i, j], [j, i]]
        escores += [[s], [s]]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)
edge_attr  = torch.tensor(escores, dtype=torch.float).to(DEVICE)

# ── Модел ────────────────────────────────────────────────────────────────────
model = BioGNN(1, 128, len(CLASSES), edge_dim=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# ── Предсказания (същата нормализация като backend) ──────────────────────────
y_true, y_pred = [], []
batch = torch.zeros(len(gene_list), dtype=torch.long).to(DEVICE)
for p in common:
    vals = expr[p].reindex(gene_list).fillna(0.0).values.astype(float)
    x = torch.tensor(vals, dtype=torch.float).view(-1, 1).to(DEVICE)
    x = (x - x.mean()) / (x.std() + 1e-8)
    with torch.no_grad():
        logits = model(x, edge_index, batch, edge_attr=edge_attr)
        pred = int(logits.argmax(dim=1).item())
    y_pred.append(CLASSES[pred])
    y_true.append(labels[p])

y_true = np.array(y_true); y_pred = np.array(y_pred)

print("\n===== ТОЧНОСТ =====")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1 : {f1_score(y_true, y_pred, average='macro', labels=CLASSES):.4f}")

print("\n===== PER-CLASS =====")
print(classification_report(y_true, y_pred, labels=CLASSES, digits=4, zero_division=0))

print("\n===== CONFUSION MATRIX (ред=истински, колона=предсказан) =====")
cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
hdr = "истински \\ предсказан".ljust(16) + "".join(c.replace('BRCA_','').rjust(8) for c in CLASSES) + "   total"
print(hdr)
for i, c in enumerate(CLASSES):
    row = c.ljust(16) + "".join(str(cm[i, j]).rjust(8) for j in range(len(CLASSES)))
    print(row + str(cm[i].sum()).rjust(8))

print("\n===== НАЙ-ГОЛЕМИ ОБЪРКВАНИЯ =====")
mis = []
for i, ci in enumerate(CLASSES):
    for j, cj in enumerate(CLASSES):
        if i != j and cm[i, j] > 0:
            pct = 100 * cm[i, j] / cm[i].sum()
            mis.append((cm[i, j], pct, ci, cj))
for cnt, pct, ci, cj in sorted(mis, reverse=True)[:8]:
    print(f"  {ci:14s} -> {cj:14s}: {cnt:3d} пациента ({pct:.1f}% от {ci})")
