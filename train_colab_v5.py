"""
================================================================================
train_colab_v5.py — BioGNNv4 + FocalLoss(γ=1.0) + DropEdge
================================================================================
Цел: поправи LumA recall (0.54→~0.75) без да губим Her2/LumB напредъка

ПРОМЕНИ СПРЯМО v4 (Acc=0.6932, F1=0.6646):
─────────────────────────────────────────────
  [1] FOCAL_GAMMA: 2.0 → 1.0
      v4 с γ=2.0 беше прекалено агресивен — жертва LumA за да спаси Normal/Her2.
      γ=1.0 е баланс: все още фокусира върху трудни примери, но по-нежно.

  [2] DropEdge(p=0.15) при тренировка
      При всяка batch стъпка случайно премахва 15% от PPI ребрата.
      Моделът се учи да не разчита на конкретни взаимодействия → по-добра
      генерализация. Само при тренировка — при eval всички ребра са активни.

  Всичко останало от v4 е запазено:
    • 3 node features (z_score, degree_norm, mean_edge_weight)
    • BioGNNv4: 3 GATv2 слоя + JumpingKnowledge + dual readout [mean‖max]
    • AdamW + CosineAnnealingWarmRestarts(T_0=60)
    • Gradient clipping (max_norm=1.0)
    • EPOCHS=300, PATIENCE=40

КАК ДА ПУСНЕШ В COLAB (T4 GPU):
──────────────────────────────────
  %run train_colab_v5.py
  Очаквано: ~70–100 мин на T4.
  Изход: best_model_v5_196genes.pt + v5_results.txt
================================================================================
"""

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree as pyg_degree, dropout_edge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

BASE      = '.'
EXPR_PATH = os.path.join(BASE, 'data', 'tcga_expression_198genes.csv')
CLIN_PATH = os.path.join(BASE, 'data', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
PPI_PATH  = os.path.join(BASE, 'data', 'string_ppi_196genes.tsv')
CKPT_PATH = os.path.join(BASE, 'best_model_v5_196genes.pt')
OUT_PATH  = os.path.join(BASE, 'v5_results.txt')

# ── Хиперпараметри ────────────────────────────────────────────────────────────
HIDDEN      = 128
HEADS       = 4
EPOCHS      = 300
BATCH       = 16
LR          = 0.0005
WD          = 1e-4
PATIENCE    = 40
FOCAL_GAMMA = 1.0    # [ПРОМЯНА] беше 2.0 → сега 1.0 за по-мек фокус
DROP_EDGE_P = 0.15   # [ПРОМЯНА] DropEdge вероятност при тренировка
T0          = 60
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── FocalLoss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ── BioGNNv4 (идентична на v4 — само тренировката се различава) ──────────────
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


# ── Данни ─────────────────────────────────────────────────────────────────────
print("Зареждане на данни...")
clin   = pd.read_csv(CLIN_PATH, sep='\t', comment='#').dropna(subset=['Subtype'])
expr   = pd.read_csv(EXPR_PATH, index_col=0)
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr   = expr[common].fillna(0)
clin   = clin.set_index('Sample ID').loc[common, ['Subtype']]
expr   = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)
ppi    = pd.read_csv(PPI_PATH, sep='\t')

le      = LabelEncoder()
y_enc   = le.fit_transform(clin['Subtype'])
classes = list(le.classes_)
N, NC   = len(expr), len(classes)
print(f"Пациенти: {len(common)} | Гени: {N} | Класове: {classes}")

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
print(f"Ребра: {base_ei.shape[1]} | edge_attr: [{base_ea.min():.3f}, {base_ea.max():.3f}]")

# ── Структурни node features ──────────────────────────────────────────────────
deg_raw  = pyg_degree(base_ei[0], num_nodes=N).float()
deg_norm = deg_raw / (deg_raw.max() + 1e-8)
mean_ew  = torch.zeros(N)
for i in range(N):
    mask = (base_ei[0] == i)
    if mask.sum() > 0:
        mean_ew[i] = base_ea[mask, 0].mean()
struct_features = torch.stack([deg_norm, mean_ew], dim=1)
IN_FEAT = 3
print(f"Node feat dim: {IN_FEAT} (z_score + degree_norm + mean_edge_weight)\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = []
for idx, pid in enumerate(expr.columns):
    z = torch.tensor(expr[pid].values, dtype=torch.float).view(-1, 1)
    x = torch.cat([z, struct_features], dim=1)
    dataset.append(Data(x=x, edge_index=base_ei, edge_attr=base_ea,
                        y=torch.tensor([y_enc[idx]], dtype=torch.long)))

counts  = clin['Subtype'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = (weights / weights.sum() * NC).to(DEVICE)
all_labels = [g.y.item() for g in dataset]


# ── Training loop ─────────────────────────────────────────────────────────────
def run_fold(train_graphs, test_graphs):
    tl  = DataLoader(train_graphs, batch_size=BATCH, shuffle=True, drop_last=True)
    vl  = DataLoader(test_graphs,  batch_size=BATCH)
    model = BioGNNv4(IN_FEAT, HIDDEN, NC, edge_dim=1, heads=HEADS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=2, eta_min=1e-6)
    crit  = FocalLoss(alpha=weights, gamma=FOCAL_GAMMA)
    best_acc, best_f1, best_state = 0.0, 0.0, None
    no_imp = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train с DropEdge ──
        model.train()
        total_loss = 0.0
        for b in tl:
            b = b.to(DEVICE)
            opt.zero_grad()

            # [ПРОМЯНА] DropEdge: 15% от ребрата се изключват при всяка стъпка
            ei_drop, edge_mask = dropout_edge(b.edge_index, p=DROP_EDGE_P, training=True)
            ea_drop = b.edge_attr[edge_mask]

            out  = model(b.x, ei_drop, b.batch, edge_attr=ea_drop)
            loss = crit(out, b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()

        sch.step()

        # ── Eval — БЕЗ DropEdge (пълният граф) ──
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(DEVICE)
                out = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr)
                ps.extend(out.argmax(1).cpu().numpy())
                ts.extend(b.y.cpu().numpy())

        acc = float(np.mean(np.array(ps) == np.array(ts)))
        f1  = f1_score(ts, ps, average='macro', zero_division=0)
        sch.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(tl):.4f} | "
                  f"Acc: {acc:.4f} | F1: {f1:.4f} | Best: {best_acc:.4f} | "
                  f"LR: {opt.param_groups[0]['lr']:.6f}")

        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds, best_trues = list(ps), list(ts)
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            print(f"  Early stopping на epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    return best_acc, best_f1, best_state, best_preds, best_trues


# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_f1s = [], []
all_preds_cv, all_trues_cv = [], []
global_best_acc, global_best_state = 0.0, None
t_start = time.time()

print(f"Стартиране на 5-Fold CV — BioGNNv4 + FocalLoss(γ={FOCAL_GAMMA}) + DropEdge({DROP_EDGE_P})")
print(f"  hidden={HIDDEN}, heads={HEADS}, in_feat={IN_FEAT}, T0={T0}\n")

for fold, (tr_idx, te_idx) in enumerate(skf.split(range(len(dataset)), all_labels), 1):
    print(f"{'='*60}\nFOLD {fold}/5\n{'='*60}")

    X_tr_z = np.array([dataset[i].x[:, 0].numpy() for i in tr_idx])
    y_tr   = np.array([dataset[i].y.item()         for i in tr_idx])
    X_aug, y_aug = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr_z, y_tr)
    print(f"  Train: {len(X_tr_z)} → {len(X_aug)} след SMOTE")

    train_graphs = []
    for x_flat, y_val in zip(X_aug, y_aug):
        z = torch.tensor(x_flat, dtype=torch.float).view(-1, 1)
        x = torch.cat([z, struct_features], dim=1)
        train_graphs.append(Data(x=x, edge_index=base_ei, edge_attr=base_ea,
                                 y=torch.tensor([y_val], dtype=torch.long)))
    test_graphs = [dataset[i] for i in te_idx]

    t0 = time.time()
    acc, f1, state, preds, trues = run_fold(train_graphs, test_graphs)
    fold_accs.append(acc); fold_f1s.append(f1)
    all_preds_cv.extend(preds); all_trues_cv.extend(trues)
    print(f"\n  Fold {fold} Best → Acc: {acc:.4f} | F1: {f1:.4f} ({(time.time()-t0)/60:.1f} мин)")
    if acc > global_best_acc:
        global_best_acc, global_best_state = acc, state
        print(f"  ★ Нов глобален best (acc={acc:.4f})")

if global_best_state:
    torch.save(global_best_state, CKPT_PATH)

total_min = (time.time() - t_start) / 60
mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
mean_f1,  std_f1  = np.mean(fold_f1s),  np.std(fold_f1s)
per_class = classification_report(all_trues_cv, all_preds_cv, target_names=classes, digits=4, zero_division=0)
cm = confusion_matrix(all_trues_cv, all_preds_cv)

lines = []
lines.append("=" * 70)
lines.append("BioGNNv4 + FocalLoss(γ=1.0) + DropEdge — ФИНАЛНИ РЕЗУЛТАТИ (v5)")
lines.append("=" * 70)
lines.append(f"  FocalLoss gamma: {FOCAL_GAMMA}  |  DropEdge p: {DROP_EDGE_P}")
lines.append(f"  Accuracy:  {mean_acc:.4f}  (± {std_acc:.4f})")
lines.append(f"  Macro F1:  {mean_f1:.4f}  (± {std_f1:.4f})")
lines.append(f"  Fold breakdown:")
for k, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
    lines.append(f"    Fold {k}: acc={a:.4f}  f1={f:.4f}")
lines.append("")
lines.append("СРАВНЕНИЕ:")
lines.append(f"  {'Версия':<42} {'Acc':>8}  {'F1':>8}")
lines.append(f"  {'-'*60}")
lines.append(f"  {'GATv2 196г (09.06 — предишен best)':<42} {'0.6861':>8}  {'0.5429':>8}")
lines.append(f"  {'BioGNNv4 γ=2.0 (17.06)':<42} {'0.6932':>8}  {'0.6646':>8}")
lines.append(f"  {'BioGNNv4 γ=1.0 + DropEdge (18.06)':<42} {mean_acc:>8.4f}  {mean_f1:>8.4f}")
lines.append(f"  {'-'*60}")
lines.append(f"  {'RandomForest (baseline)':<42} {'0.9072':>8}  {'0.7896':>8}")
lines.append("")
lines.append("Per-class report:")
lines.append(per_class)
lines.append("Confusion matrix:")
lines.append("           " + "  ".join(f"{c[5:]:>6}" for c in classes))
for i, row in enumerate(cm):
    lines.append(f"{classes[i][5:]:>10} " + "  ".join(f"{v:>6}" for v in row))
lines.append(f"\nВреме: {total_min:.1f} мин | Checkpoint: {CKPT_PATH}")

report = "\n".join(lines)
print("\n" + report)
with open(OUT_PATH, 'w', encoding='utf-8') as fh:
    fh.write(report)
print(f"\nЗапазено: {OUT_PATH}\nCheckpoint: {CKPT_PATH} (best acc={global_best_acc:.4f})")
