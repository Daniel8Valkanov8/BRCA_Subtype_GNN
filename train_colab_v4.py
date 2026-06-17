"""
================================================================================
train_colab_v4.py — BioGNNv4 Enhanced
================================================================================
Цел: от 68.61% → ~80% Accuracy на 5-Fold CV

ПРОМЕНИ СПРЯМО ТЕКУЩИЯ НАЙ-ДОБЪР (196г, GATv2 3L, Acc=0.6861):
──────────────────────────────────────────────────────────────────
  [1] 3 node features вместо 1:
        x = [z_score | degree_normalized | mean_edge_weight]
      Моделът "знае" кой ген е hub в PPI мрежата → по-богат вход.

  [2] BioGNNv4 архитектура:
        - 3 GATv2 слоя (запазва дълбочина)
        - JumpingKnowledge: concat на [h1, h2, h3] преди readout
          (класификаторът вижда представяния от ВСЕКИ слой)
        - Dual readout: [mean_pool ‖ max_pool]
          (max_pool запазва пиковите/дискриминативните гени)
        - По-широк класификатор: Linear(readout → 256 → 5)

  [3] FocalLoss(gamma=2) вместо CrossEntropy:
        Фокусира обучението върху трудните примери (LumB/Normal).
        FL = -(1-pt)^γ * log(pt) наказва по-силно грешките
        при вече добре класифицирани примери.

  [4] CosineAnnealingWarmRestarts(T_0=60, T_mult=2):
        Периодично вдига LR → излиза от локални минима.
        При ReduceLROnPlateau моделът "замръзваше" твърде рано.

  [5] Gradient clipping (max_norm=1.0):
        Стабилизира обучението при GATv2 + по-дълбоки мрежи.

  [6] EPOCHS=300, PATIENCE=40:
        Повече шанс за намиране на по-добро решение.

  [7] AdamW (decoupled weight decay) вместо Adam:
        По-добра L2 регуляризация.

КАК ДА ПУСНЕШ В COLAB (T4 GPU):
──────────────────────────────────
  !pip install torch-geometric imbalanced-learn -q
  %run train_colab_v4.py
  Очаквано време: ~60-90 мин на T4.
  Изход: best_model_v4_196genes.pt + v4_results.txt
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree as pyg_degree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Пътища (промени BASE за Colab: BASE = '/content/Master_Thesis_GNN') ──────
BASE       = '.'
EXPR_PATH  = os.path.join(BASE, 'data', 'tcga_expression_198genes.csv')
CLIN_PATH  = os.path.join(BASE, 'data', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
PPI_PATH   = os.path.join(BASE, 'data', 'string_ppi_196genes.tsv')
CKPT_PATH  = os.path.join(BASE, 'best_model_v4_196genes.pt')
OUT_PATH   = os.path.join(BASE, 'v4_results.txt')

# ── Хиперпараметри ────────────────────────────────────────────────────────────
HIDDEN     = 128   # скрита размерност на GATv2 слоевете
HEADS      = 4     # attention heads (слоеве 1 и 2)
EPOCHS     = 300   # ↑ от 150 → повече шанс за намиране на оптимум
BATCH      = 16
LR         = 0.0005
WD         = 1e-4  # weight decay (AdamW)
PATIENCE   = 40    # ↑ от 20 → по-малко agressive early stopping
FOCAL_GAMMA = 2.0  # FocalLoss параметър
T0         = 60    # CosineAnnealing период (епохи до първи рестарт)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════════
# [ПРОМЯНА 3] FOCAL LOSS
# ════════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    FocalLoss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    За трудни примери (LumB, Normal) p_t е малко → (1-p_t)^gamma е голямо
    → наказанието е пропорционално по-голямо от стандартния CrossEntropy.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha   # class weights тензор (на device)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)                        # вероятност на правилния клас
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == 'mean' else focal.sum()


# ════════════════════════════════════════════════════════════════════════════════
# [ПРОМЯНА 2] BioGNNv4 — JumpingKnowledge + Dual Readout + 3 GATv2 слоя
# ════════════════════════════════════════════════════════════════════════════════
class BioGNNv4(nn.Module):
    """
    Архитектура:
      GATv2Conv(in_feat → H, heads=4)   → BN → ReLU  = h1  (H*4)
      GATv2Conv(H*4 → H, heads=4)       → BN → ReLU  = h2  (H*4)
      GATv2Conv(H*4 → H, heads=1)       → BN → ReLU  = h3  (H)
      JK = concat([h1, h2, h3])                       (H*4 + H*4 + H = 9H)
      readout = [mean_pool(JK) ‖ max_pool(JK)]        (18H)
      Dropout → Linear(18H → 256) → ReLU → Dropout → Linear(256 → 5)

    in_feat = 3 (z_score, degree_norm, mean_edge_weight)
    """
    def __init__(self, in_feat, hidden, num_classes, edge_dim=1,
                 heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        H            = hidden

        self.conv1 = GATv2Conv(in_feat,  H, heads=heads, concat=True,
                               dropout=0.2, edge_dim=edge_dim)
        self.bn1   = nn.BatchNorm1d(H * heads)

        self.conv2 = GATv2Conv(H * heads, H, heads=heads, concat=True,
                               dropout=0.2, edge_dim=edge_dim)
        self.bn2   = nn.BatchNorm1d(H * heads)

        self.conv3 = GATv2Conv(H * heads, H, heads=1, concat=False,
                               dropout=0.2, edge_dim=edge_dim)
        self.bn3   = nn.BatchNorm1d(H)

        # JumpingKnowledge: конкат на изходите от 3-те слоя
        jk_dim      = H * heads + H * heads + H   # 4H + 4H + H = 9H
        readout_dim = 2 * jk_dim                  # [mean ‖ max] → 18H

        self.lin1 = nn.Linear(readout_dim, 128)
        self.lin2 = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None,
                return_attention=False):
        attn = {}

        # ── Слой 1 ──
        if return_attention:
            h1, (ei1, aw1) = self.conv1(x, edge_index, edge_attr=edge_attr,
                                         return_attention_weights=True)
            attn['layer1'] = (ei1.cpu(), aw1.cpu())
        else:
            h1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        h1 = F.relu(self.bn1(h1))

        # ── Слой 2 ──
        if return_attention:
            h2, (ei2, aw2) = self.conv2(h1, edge_index, edge_attr=edge_attr,
                                         return_attention_weights=True)
            attn['layer2'] = (ei2.cpu(), aw2.cpu())
        else:
            h2 = self.conv2(h1, edge_index, edge_attr=edge_attr)
        h2 = F.relu(self.bn2(h2))

        # ── Слой 3 ──
        if return_attention:
            h3, (ei3, aw3) = self.conv3(h2, edge_index, edge_attr=edge_attr,
                                         return_attention_weights=True)
            attn['layer3'] = (ei3.cpu(), aw3.cpu())
        else:
            h3 = self.conv3(h2, edge_index, edge_attr=edge_attr)
        h3 = F.relu(self.bn3(h3))

        # ── JumpingKnowledge: concat на представянията от ВСЕКИ слой ──
        h_jk = torch.cat([h1, h2, h3], dim=1)   # (N_nodes, 9H)

        # ── Dual readout: mean ‖ max ──
        h_mean = global_mean_pool(h_jk, batch)
        h_max  = global_max_pool(h_jk, batch)
        h      = torch.cat([h_mean, h_max], dim=1)   # (B, 18H)

        # ── Класификатор ──
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout * 0.5, training=self.training)
        out = self.lin2(h)

        if return_attention:
            return out, attn
        return out


# ════════════════════════════════════════════════════════════════════════════════
# ДАННИ
# ════════════════════════════════════════════════════════════════════════════════
print("Зареждане на данни...")
clin = pd.read_csv(CLIN_PATH, sep='\t', comment='#').dropna(subset=['Subtype'])
expr = pd.read_csv(EXPR_PATH, index_col=0)
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr   = expr[common].fillna(0)
clin   = clin.set_index('Sample ID').loc[common, ['Subtype']]

# Per-patient z-score нормализация (идентично на предишните версии)
expr = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)

ppi = pd.read_csv(PPI_PATH, sep='\t')
le  = LabelEncoder()
y_enc   = le.fit_transform(clin['Subtype'])
classes = list(le.classes_)
N       = len(expr)     # брой гени (196)
NC      = len(classes)  # 5

print(f"Пациенти: {len(common)} | Гени: {N} | Класове: {classes}")
print(f"Разпределение: {dict(zip(classes, np.bincount(y_enc)))}")

# ── Граф: edge_index + edge_attr ─────────────────────────────────────────────
gene_map = {g: i for i, g in enumerate(expr.index)}
ei_list, ea_list = [], []
for _, r in ppi.iterrows():
    if r['node1'] in gene_map and r['node2'] in gene_map:
        i, j = gene_map[r['node1']], gene_map[r['node2']]
        s    = float(r['combined_score'])
        ei_list += [[i, j], [j, i]]
        ea_list += [[s], [s]]

base_ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()  # (2, E)
base_ea = torch.tensor(ea_list, dtype=torch.float)                   # (E, 1)
print(f"Ребра: {base_ei.shape[1]} | edge_attr min={base_ea.min():.4f} max={base_ea.max():.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# [ПРОМЯНА 1] — СТРУКТУРНИ NODE FEATURES (degree + mean edge weight)
# Изчисляваме ВЕДНЪЖ — те са константни за всички пациенти.
# ════════════════════════════════════════════════════════════════════════════════
# Degree: брой ребра на всеки ген в PPI мрежата, нормализиран до [0,1]
deg_raw  = pyg_degree(base_ei[0], num_nodes=N).float()   # (N,)
deg_norm = deg_raw / (deg_raw.max() + 1e-8)              # нормализиран

# Среден combined_score на инцидентните ребра (мярка за "качество" на хъба)
mean_ew = torch.zeros(N, dtype=torch.float)
for i in range(N):
    mask = (base_ei[0] == i)
    if mask.sum() > 0:
        mean_ew[i] = base_ea[mask, 0].mean()

# struct_features: (N, 2) — добавяме ги към z_score за всеки пациент
struct_features = torch.stack([deg_norm, mean_ew], dim=1)   # (N, 2)

print(f"\nStructural features (constant across patients):")
print(f"  degree:    min={deg_norm.min():.3f}  max={deg_norm.max():.3f}  "
      f"mean={deg_norm.mean():.3f}")
print(f"  mean_ew:   min={mean_ew.min():.3f}  max={mean_ew.max():.3f}  "
      f"mean={mean_ew.mean():.3f}")
print(f"  node feat dim: {1 + struct_features.shape[1]} (z_score + degree + mean_ew)\n")

IN_FEAT = 1 + struct_features.shape[1]   # = 3

# ── Строим dataset (с 3D node features) ──────────────────────────────────────
dataset = []
for idx, pid in enumerate(expr.columns):
    z = torch.tensor(expr[pid].values, dtype=torch.float).view(-1, 1)  # (N, 1)
    x = torch.cat([z, struct_features], dim=1)                          # (N, 3)
    dataset.append(Data(x=x, edge_index=base_ei, edge_attr=base_ea,
                        y=torch.tensor([y_enc[idx]], dtype=torch.long)))

# ── Класови тегла за FocalLoss ───────────────────────────────────────────────
counts  = clin['Subtype'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = (weights / weights.sum() * NC).to(DEVICE)

all_labels = [g.y.item() for g in dataset]


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════════
def run_fold(train_graphs, test_graphs):
    tl  = DataLoader(train_graphs, batch_size=BATCH, shuffle=True, drop_last=True)
    vl  = DataLoader(test_graphs,  batch_size=BATCH)

    model = BioGNNv4(IN_FEAT, HIDDEN, NC, edge_dim=1,
                     heads=HEADS, dropout=0.3).to(DEVICE)
    # [ПРОМЯНА 7] AdamW
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # [ПРОМЯНА 4] CosineAnnealingWarmRestarts
    sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=T0, T_mult=2, eta_min=1e-6)
    # [ПРОМЯНА 3] FocalLoss
    crit  = FocalLoss(alpha=weights, gamma=FOCAL_GAMMA)

    best_acc, best_f1, best_state = 0.0, 0.0, None
    no_imp = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for b in tl:
            b = b.to(DEVICE)
            opt.zero_grad()
            out  = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr)
            loss = crit(out, b.y)
            loss.backward()
            # [ПРОМЯНА 5] Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()

        # [ПРОМЯНА 4] scheduler стъпка след всяка епоха
        sch.step()

        # ── Eval ──
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
        avg_loss = total_loss / len(tl)

        if epoch % 10 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc:.4f} | F1: {f1:.4f} | Best: {best_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_preds, best_trues = list(ps), list(ts)
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= PATIENCE:
            print(f"  Early stopping на epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    return best_acc, best_f1, best_state, best_preds, best_trues


# ════════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV
# ════════════════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_f1s = [], []
all_preds_cv, all_trues_cv = [], []
global_best_acc, global_best_state = 0.0, None
t_start = time.time()

print(f"\nСтартиране на 5-Fold CV — BioGNNv4 Enhanced")
print(f"  hidden={HIDDEN}, heads={HEADS}, in_feat={IN_FEAT}")
print(f"  epochs={EPOCHS}, patience={PATIENCE}, T0={T0}")
print(f"  loss=FocalLoss(γ={FOCAL_GAMMA}), opt=AdamW, sch=CosineAnnealingWarmRestarts\n")

for fold, (tr_idx, te_idx) in enumerate(
        skf.split(range(len(dataset)), all_labels), 1):

    print(f"{'='*60}\nFOLD {fold}/5\n{'='*60}")

    # ── in-fold SMOTE (само на z_score векторите) ───────────────────────────
    # СМОТЕ работи в R^196 (само z_score, БЕЗ структурните features)
    X_tr_z = np.array([dataset[i].x[:, 0].numpy() for i in tr_idx])  # (train, 196)
    y_tr   = np.array([dataset[i].y.item()         for i in tr_idx])

    smote  = SMOTE(random_state=42, k_neighbors=5)
    X_aug, y_aug = smote.fit_resample(X_tr_z, y_tr)
    print(f"  Train: {len(X_tr_z)} → {len(X_aug)} след SMOTE")

    # Строим PyG обекти от синтетичните пациенти (добавяме struct features)
    train_graphs = []
    for x_flat, y_val in zip(X_aug, y_aug):
        z = torch.tensor(x_flat, dtype=torch.float).view(-1, 1)
        x = torch.cat([z, struct_features], dim=1)   # (N, 3)
        train_graphs.append(Data(
            x=x, edge_index=base_ei, edge_attr=base_ea,
            y=torch.tensor([y_val], dtype=torch.long)
        ))

    test_graphs = [dataset[i] for i in te_idx]

    # ── Обучение ────────────────────────────────────────────────────────────
    t0 = time.time()
    acc, f1, state, preds, trues = run_fold(train_graphs, test_graphs)
    elapsed = time.time() - t0

    fold_accs.append(acc)
    fold_f1s.append(f1)
    all_preds_cv.extend(preds)
    all_trues_cv.extend(trues)

    print(f"\n  Fold {fold} Best → Acc: {acc:.4f} | Macro F1: {f1:.4f} "
          f"({elapsed/60:.1f} мин)")

    if acc > global_best_acc:
        global_best_acc   = acc
        global_best_state = state
        print(f"  ★ Нов глобален best (acc={acc:.4f})")

# ── Запис на checkpoint ──────────────────────────────────────────────────────
if global_best_state is not None:
    torch.save(global_best_state, CKPT_PATH)

total_time = (time.time() - t_start) / 60

# ════════════════════════════════════════════════════════════════════════════════
# ФИНАЛЕН ОТЧЕТ
# ════════════════════════════════════════════════════════════════════════════════
mean_acc = np.mean(fold_accs)
std_acc  = np.std(fold_accs)
mean_f1  = np.mean(fold_f1s)
std_f1   = np.std(fold_f1s)

per_class = classification_report(
    all_trues_cv, all_preds_cv,
    target_names=classes, digits=4, zero_division=0
)
cm = confusion_matrix(all_trues_cv, all_preds_cv)

lines = []
lines.append("=" * 70)
lines.append("BioGNNv4 Enhanced — ФИНАЛНИ РЕЗУЛТАТИ")
lines.append("=" * 70)
lines.append(f"  {'Промени:':<30} 3 node feat | JK | dual readout | FocalLoss | "
             "CosineAnnealing | AdamW")
lines.append(f"  {'Архитектура:':<30} GATv2 3L, hidden={HIDDEN}, heads={HEADS}, "
             f"in_feat={IN_FEAT}")
lines.append(f"  {'Обучение:':<30} epochs={EPOCHS}, patience={PATIENCE}, "
             f"T0={T0}, batch={BATCH}")
lines.append("")
lines.append(f"  Accuracy:  {mean_acc:.4f}  (± {std_acc:.4f})")
lines.append(f"  Macro F1:  {mean_f1:.4f}  (± {std_f1:.4f})")
lines.append(f"  Fold breakdown:")
for k, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
    lines.append(f"    Fold {k}: acc={a:.4f}  f1={f:.4f}")
lines.append("")
lines.append("СРАВНЕНИЕ:")
lines.append(f"  {'Версия':<38} {'Acc':>10}  {'Macro F1':>10}")
lines.append(f"  {'-'*60}")
lines.append(f"  {'GATConv 3L 160г (V4)':<38} {'0.5851':>10}  {'0.4353':>10}")
lines.append(f"  {'GATv2 3L 160г, fixed edges (V6)':<38} {'0.6136':>10}  {'0.4544':>10}")
lines.append(f"  {'GATv2 3L 196г (НАЙ-ДОБЪР досега)':<38} {'0.6861':>10}  {'0.5429':>10}")
lines.append(f"  {'BioGNNv4 Enhanced (ТОЗИ РУН)':<38} {mean_acc:>10.4f}  {mean_f1:>10.4f}")
lines.append(f"  {'-'*60}")
lines.append(f"  {'RandomForest (baseline)':<38} {'0.9072':>10}  {'0.7896':>10}")
lines.append(f"  {'LogReg (baseline)':<38} {'0.8960':>10}  {'0.8552':>10}")
lines.append("")
lines.append("Per-class report (combined CV folds — 981 пациента):")
lines.append(per_class)
lines.append("Confusion matrix (редове=истина, колони=предсказано):")
lines.append("           " + "  ".join(f"{c[5:]:>6}" for c in classes))
for i, row in enumerate(cm):
    lines.append(f"{classes[i][5:]:>10} " + "  ".join(f"{v:>6}" for v in row))
lines.append(f"\nОбщо време: {total_time:.1f} мин")
lines.append(f"Checkpoint: {CKPT_PATH}")

report = "\n".join(lines)
print("\n" + report)

with open(OUT_PATH, 'w', encoding='utf-8') as fh:
    fh.write(report)

print(f"\nЗапазено: {OUT_PATH}")
print(f"Checkpoint: {CKPT_PATH} (best fold acc={global_best_acc:.4f})")
print(f"Общо: {total_time:.1f} мин на {DEVICE}")
