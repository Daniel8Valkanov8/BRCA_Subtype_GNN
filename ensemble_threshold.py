"""
================================================================================
ensemble_threshold.py — Ensemble + Threshold оптимизация
================================================================================
Зарежда v4 и v5 модели (едни и същи BioGNNv4 архитектура, 3 node features),
усреднява softmax вероятностите и оптимизира decision thresholds per class.

Очаквано подобрение спрямо v4 сам:
  Acc:  0.6932 → ~0.73–0.76
  F1:   0.6646 → ~0.68–0.73

ИЗИСКВАНИЯ (Colab):
  - best_model_v5_196genes.pt  (или model_18-06-26.pt  — нов модел, точно обучен)
  - last_model-17-06.pt        (или best_model_v4_196genes.pt — стар v4)
  - данните: data/*.csv / *.tsv
================================================================================
"""
import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import degree as pyg_degree
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ── Пътища ───────────────────────────────────────────────────────────────────
BASE      = '.'
EXPR_PATH = os.path.join(BASE, 'data', 'tcga_expression_198genes.csv')
CLIN_PATH = os.path.join(BASE, 'data', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
PPI_PATH  = os.path.join(BASE, 'data', 'string_ppi_196genes.tsv')

# Кои модели да заредим
V4_CANDIDATES = ['last_model-17-06.pt', 'best_model_v4_196genes.pt']
V5_CANDIDATES = ['best_model_v5_196genes.pt', 'model_18-06-26.pt']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── Архитектура (идентична на двата скрипта) ─────────────────────────────────
class BioGNNv4(nn.Module):
    def __init__(self, in_feat, hidden, num_classes, edge_dim=1, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        H = hidden
        self.conv1 = GATv2Conv(in_feat,   H, heads=heads, concat=True,  dropout=0.2, edge_dim=edge_dim)
        self.bn1   = nn.BatchNorm1d(H * heads)
        self.conv2 = GATv2Conv(H * heads, H, heads=heads, concat=True,  dropout=0.2, edge_dim=edge_dim)
        self.bn2   = nn.BatchNorm1d(H * heads)
        self.conv3 = GATv2Conv(H * heads, H, heads=1,     concat=False, dropout=0.2, edge_dim=edge_dim)
        self.bn3   = nn.BatchNorm1d(H)
        jk_dim      = H * heads + H * heads + H
        readout_dim = 2 * jk_dim
        self.lin1 = nn.Linear(readout_dim, 128)
        self.lin2 = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index, edge_attr=edge_attr)))
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index, edge_attr=edge_attr)))
        h_jk = torch.cat([h1, h2, h3], dim=1)
        h = torch.cat([global_mean_pool(h_jk, batch),
                       global_max_pool(h_jk, batch)], dim=1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout * 0.5, training=self.training)
        return self.lin2(h)


# ── Зареждане на данни ────────────────────────────────────────────────────────
print("\nЗареждане на данни...")
clin   = pd.read_csv(CLIN_PATH, sep='\t', comment='#').dropna(subset=['Subtype'])
expr   = pd.read_csv(EXPR_PATH, index_col=0)
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr   = expr[common].fillna(0)
clin   = clin.set_index('Sample ID').loc[common, ['Subtype']]
expr   = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)
ppi    = pd.read_csv(PPI_PATH, sep='\t')

le      = LabelEncoder()
y_true  = le.fit_transform(clin['Subtype'])
classes = list(le.classes_)
NC      = len(classes)
N       = len(expr)
print(f"  {len(common)} пациента | {N} гена | {NC} класа")

gene_map = {g: i for i, g in enumerate(expr.index)}
ei_list, ea_list = [], []
for _, r in ppi.iterrows():
    if r['node1'] in gene_map and r['node2'] in gene_map:
        i, j = gene_map[r['node1']], gene_map[r['node2']]
        s = float(r['combined_score'])
        ei_list += [[i, j], [j, i]]
        ea_list += [[s], [s]]
base_ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
base_ea = torch.tensor(ea_list, dtype=torch.float)

deg_raw  = pyg_degree(base_ei[0], num_nodes=N).float()
deg_norm = deg_raw / (deg_raw.max() + 1e-8)
mean_ew  = torch.zeros(N)
for i in range(N):
    mask = (base_ei[0] == i)
    if mask.sum() > 0:
        mean_ew[i] = base_ea[mask, 0].mean()
struct_features = torch.stack([deg_norm, mean_ew], dim=1)
IN_FEAT = 3

dataset = []
for idx, pid in enumerate(expr.columns):
    z = torch.tensor(expr[pid].values, dtype=torch.float).view(-1, 1)
    x = torch.cat([z, struct_features], dim=1)
    dataset.append(Data(x=x, edge_index=base_ei, edge_attr=base_ea,
                        y=torch.tensor([y_true[idx]], dtype=torch.long)))
loader = DataLoader(dataset, batch_size=32, shuffle=False)


# ── Utility: inference → probability matrix ───────────────────────────────────
def get_probs(model_path):
    model = BioGNNv4(IN_FEAT, 128, NC, edge_dim=1, heads=4).to(DEVICE)
    sd = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            logits = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr)
            probs_list.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs_list, dim=0).numpy()   # (981, 5)


def report_model(name, probs, y, classes):
    preds = probs.argmax(axis=1)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average='macro', zero_division=0)
    print(f"\n{'─'*60}")
    print(f"{name}")
    print(f"  Accuracy:  {acc:.4f}  |  Macro F1:  {f1:.4f}")
    print(classification_report(y, preds, target_names=classes, digits=4, zero_division=0))
    return acc, f1, preds


def predict_with_thresholds(probs_np, thresholds):
    """Нормализирани вероятности по threshold → argmax"""
    scaled = probs_np / np.array(thresholds)
    return scaled.argmax(axis=1)


def grid_search_thresholds(probs_np, y, classes):
    """Пълен grid search: оптимизира thresholds за LumA и LumB"""
    NC = len(classes)
    luma_idx  = classes.index('BRCA_LumA')
    lumb_idx  = classes.index('BRCA_LumB')
    norm_idx  = classes.index('BRCA_Normal')

    best_f1, best_t = 0.0, None
    base = [0.45] * NC                       # start: все равни прагове
    base[classes.index('BRCA_Basal')] = 0.45

    for luma_t in np.arange(0.25, 0.55, 0.025):    # намаляваме → повече LumA
        for lumb_t in np.arange(0.35, 0.70, 0.025):  # вдигаме → по-малко LumB
            for norm_t in np.arange(0.30, 0.55, 0.025):
                t = list(base)
                t[luma_idx] = luma_t
                t[lumb_idx] = lumb_t
                t[norm_idx] = norm_t
                preds = predict_with_thresholds(probs_np, t)
                f1 = f1_score(y, preds, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t[:]

    return best_t, best_f1


# ── Намиране на модели ────────────────────────────────────────────────────────
def find_model(candidates):
    for c in candidates:
        p = os.path.join(BASE, c)
        if os.path.exists(p):
            return p
    return None

v4_path = find_model(V4_CANDIDATES)
v5_path = find_model(V5_CANDIDATES)
print(f"\nМодели:")
print(f"  v4: {v4_path or 'НЕ НАМЕРЕН'}")
print(f"  v5: {v5_path or 'НЕ НАМЕРЕН (стартирай тренировката)'}")

available = [p for p in [v4_path, v5_path] if p]
if not available:
    raise FileNotFoundError("Нито един модел не е намерен. Пусни тренировката или качи .pt файловете.")

# ── Inference ─────────────────────────────────────────────────────────────────
all_probs = {}
print("\nInference...")
if v4_path:
    all_probs['v4 (γ=2.0, 17.06)'] = get_probs(v4_path)
    print(f"  v4 готов")
if v5_path:
    all_probs['v5 (γ=1.0, DropEdge)'] = get_probs(v5_path)
    print(f"  v5 готов")

# ── Резултати — отделни модели ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИНДИВИДУАЛНИ РЕЗУЛТАТИ")
print("=" * 60)
for name, probs in all_probs.items():
    report_model(name, probs, y_true, classes)

# ── Ensemble ──────────────────────────────────────────────────────────────────
if len(all_probs) == 2:
    probs_list = list(all_probs.values())
    ens_probs  = np.mean(probs_list, axis=0)
    print("\n" + "=" * 60)
    print("ENSEMBLE (средна стойност на вероятностите)")
    print("=" * 60)
    ens_acc, ens_f1, _ = report_model("Ensemble v4+v5", ens_probs, y_true, classes)
elif len(all_probs) == 1:
    print("\n[INFO] Само един модел е намерен — ще оптимизираме thresholds за него")
    ens_probs = list(all_probs.values())[0]
    ens_acc   = accuracy_score(y_true, ens_probs.argmax(axis=1))
    ens_f1    = f1_score(y_true, ens_probs.argmax(axis=1), average='macro', zero_division=0)
else:
    ens_probs = list(all_probs.values())[0]

# ── Threshold оптимизация ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("THRESHOLD ОПТИМИЗАЦИЯ (grid search на ensemble вероятности)")
print("[!] Забележка: thresholds са намерени IN-SAMPLE — истинско подобрение")
print("    ще се потвърди при следващо CV обучение с тези thresholds.")
print("=" * 60)
print("Търсене... (може да отнеме 30–60 сек)")

best_t, best_f1_thresh = grid_search_thresholds(ens_probs, y_true, classes)
thresh_preds = predict_with_thresholds(ens_probs, best_t)
thresh_acc   = accuracy_score(y_true, thresh_preds)
thresh_f1    = f1_score(y_true, thresh_preds, average='macro', zero_division=0)

print(f"\nОптимални thresholds:")
for c, t in zip(classes, best_t):
    print(f"  {c[5:]:>8}: {t:.3f}")

print(f"\n{'─'*60}")
print(f"Ensemble + Threshold оптимизация:")
print(f"  Accuracy: {thresh_acc:.4f}")
print(f"  Macro F1: {thresh_f1:.4f}")
print(classification_report(y_true, thresh_preds, target_names=classes, digits=4, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, thresh_preds)
print("Confusion matrix (ensemble + thresholds):")
short = [c[5:] for c in classes]
print("         " + "  ".join(f"{s:>7}" for s in short))
for i, row in enumerate(cm):
    print(f"{short[i]:>8} " + "  ".join(f"{v:>7}" for v in row))

# ── Финално сравнение ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ФИНАЛНО СРАВНЕНИЕ")
print(f"{'='*60}")
print(f"  {'Метод':<42} {'Acc':>8}  {'F1':>8}")
print(f"  {'-'*60}")
print(f"  {'GATv2 196г (09.06, стар best)':<42} {'0.6861':>8}  {'0.5429':>8}")
print(f"  {'BioGNNv4 v4 (γ=2.0, 17.06)':<42} {'0.6932':>8}  {'0.6646':>8}")
for name, probs in all_probs.items():
    a = accuracy_score(y_true, probs.argmax(axis=1))
    f = f1_score(y_true, probs.argmax(axis=1), average='macro', zero_division=0)
    print(f"  {name:<42} {a:>8.4f}  {f:>8.4f}")
if len(all_probs) == 2:
    print(f"  {'Ensemble v4+v5':<42} {ens_acc:>8.4f}  {ens_f1:>8.4f}")
print(f"  {'Ensemble + Threshold оптимизация':<42} {thresh_acc:>8.4f}  {thresh_f1:>8.4f}")
print(f"  {'-'*60}")
print(f"  {'RandomForest (baseline)':<42} {'0.9072':>8}  {'0.7896':>8}")
print(f"  {'LogReg (baseline)':<42} {'0.8960':>8}  {'0.8552':>8}")
print(f"{'='*60}")

# ── Запис ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'ensemble_results.txt')
with open(out_path, 'w', encoding='utf-8') as fh:
    fh.write(f"ENSEMBLE + THRESHOLD — BRCA Subtype GNN\n")
    fh.write(f"Ensemble Acc: {ens_acc:.4f} | F1: {ens_f1:.4f}\n")
    fh.write(f"Threshold Acc: {thresh_acc:.4f} | F1: {thresh_f1:.4f}\n\n")
    fh.write(f"Thresholds: {dict(zip([c[5:] for c in classes], [f'{t:.3f}' for t in best_t]))}\n\n")
    fh.write(classification_report(y_true, thresh_preds, target_names=classes, digits=4, zero_division=0))
print(f"\nЗапазено: {out_path}")
