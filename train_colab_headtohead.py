"""
================================================================================
COLAB HEAD-TO-HEAD — OLD BioGNN (текущ най-добър, 0.6861) vs NEW BioGNNv3
================================================================================
Цел: на GPU (Colab T4) да валидираме дали архитектурните корекции вдигат точността
над текущия най-добър модел (196 гена, GATv2, Acc=0.6861 / Macro F1=0.5429).

NEW корекции (адресират диагнозата "global_mean_pool заличава per-ген сигнала"):
  • dual readout [mean ‖ max]   • JumpingKnowledge (конкат на слоевете)   • 2 слоя

------------------------------------------------------------------------------
КАК ДА ПУСНЕШ В COLAB:
  1) Качи папката с data/ (или git clone) и постави този файл до нея.
  2) Първа клетка:
        !pip install torch-geometric imbalanced-learn -q
  3) Втора клетка (смени BASE ако трябва):
        %run train_colab_headtohead.py
  Очаквано време: ~30-50 мин на T4 (2 модела × 5 fold).
  Изход: head_to_head_results.txt + best_model_v3_196genes.pt
------------------------------------------------------------------------------
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Пътища (за Colab: BASE = '/content/Master_Thesis_GNN') ──
BASE      = '.'
EXPR_PATH = os.path.join(BASE, 'data', 'tcga_expression_198genes.csv')
CLIN_PATH = os.path.join(BASE, 'data', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
PPI_PATH  = os.path.join(BASE, 'data', 'string_ppi_196genes.tsv')
OUT_PATH  = os.path.join(BASE, 'head_to_head_results.txt')
CKPT_PATH = os.path.join(BASE, 'best_model_v3_196genes.pt')

# ── Хиперпараметри (идентични на най-добрия рун за честно сравнение) ──
HIDDEN, EPOCHS, BATCH, LR, WD, PATIENCE = 128, 150, 16, 5e-4, 1e-4, 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# МОДЕЛ 1 — OLD BioGNN (точно копие на текущия най-добър, src/model.py)
# ════════════════════════════════════════════════════════════════════════════
class BioGNN_OLD(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, edge_dim=1):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=4, dropout=0.2, edge_dim=edge_dim)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, dropout=0.2, edge_dim=edge_dim)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.conv3 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=1, concat=False, dropout=0.2, edge_dim=edge_dim)
        self.lin1  = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2  = torch.nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_attr)))
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.lin2(x)


# ════════════════════════════════════════════════════════════════════════════
# МОДЕЛ 2 — NEW BioGNNv3 (подобрена: dual readout + JumpingKnowledge + 2 слоя)
# ════════════════════════════════════════════════════════════════════════════
class BioGNNv3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, edge_dim=1, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=heads, dropout=0.2, edge_dim=edge_dim)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.2, edge_dim=edge_dim)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels)
        jk_dim      = hidden_channels * heads + hidden_channels   # JK конкат
        readout_dim = 2 * jk_dim                                   # [mean ‖ max]
        self.lin1 = torch.nn.Linear(readout_dim, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index, edge_attr=edge_attr)))
        h  = torch.cat([h1, h2], dim=1)                            # JumpingKnowledge
        h  = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)  # dual readout
        h  = F.dropout(h, p=self.dropout, training=self.training)
        h  = F.relu(self.lin1(h))
        h  = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)


# ════════════════════════════════════════════════════════════════════════════
# ДАННИ — 196 гена, per-patient z-score, PPI граф с edge_attr (combined_score)
# ════════════════════════════════════════════════════════════════════════════
print("Зареждане на данни...")
clin = pd.read_csv(CLIN_PATH, sep='\t', comment='#').dropna(subset=['Subtype'])
expr = pd.read_csv(EXPR_PATH, index_col=0)
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr = expr[common].fillna(0)
clin = clin.set_index('Sample ID').loc[common, ['Subtype']]
expr = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)   # per-patient z-score
ppi  = pd.read_csv(PPI_PATH, sep='\t')

le = LabelEncoder()
y_enc = le.fit_transform(clin['Subtype'])
classes = list(le.classes_)
gene_map = {g: i for i, g in enumerate(expr.index)}

# Edge index + edge_attr (combined_score), двупосочни ребра
ei_list, ea_list = [], []
for _, r in ppi.iterrows():
    if r['node1'] in gene_map and r['node2'] in gene_map:
        i, j = gene_map[r['node1']], gene_map[r['node2']]
        s = float(r['combined_score'])
        ei_list += [[i, j], [j, i]]
        ea_list += [[s], [s]]
base_ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
base_ea = torch.tensor(ea_list, dtype=torch.float)

dataset = []
for idx, pid in enumerate(expr.columns):
    x = torch.tensor(expr[pid].values, dtype=torch.float).view(-1, 1)
    dataset.append(Data(x=x, edge_index=base_ei, edge_attr=base_ea,
                        y=torch.tensor([y_enc[idx]], dtype=torch.long)))
print(f"Пациенти: {len(dataset)} | Гени: {expr.shape[0]} | Ребра: {base_ei.shape[1]} | Класове: {classes}")

# Класови тегла (преди SMOTE)
counts  = clin['Subtype'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = (weights / weights.sum() * len(counts)).to(DEVICE)
labels  = [g.y.item() for g in dataset]
NC      = len(classes)


def build(kind):
    return (BioGNN_OLD if kind == 'OLD' else BioGNNv3)(1, HIDDEN, NC, edge_dim=1).to(DEVICE)


def train_eval(kind, train_graphs, test_graphs):
    tl = DataLoader(train_graphs, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(test_graphs, batch_size=BATCH)
    model = build(kind)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    best_acc, best_f1, best_state, no_imp = 0.0, 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in tl:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr), b.y)
            loss.backward()
            opt.step()
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(DEVICE)
                out = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr)
                ps.extend(out.argmax(1).cpu().numpy()); ts.extend(b.y.cpu().numpy())
        acc = float(np.mean(np.array(ps) == np.array(ts)))
        f1  = f1_score(ts, ps, average='macro', zero_division=0)
        sch.step(acc)
        if acc > best_acc:
            best_acc, best_f1, no_imp = acc, f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds, best_trues = list(ps), list(ts)
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    return best_acc, best_f1, best_state, best_preds, best_trues


# ════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV — OLD и NEW на ИДЕНТИЧНИ folds (in-fold SMOTE)
# ════════════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {'OLD': {'acc': [], 'f1': [], 'preds': [], 'trues': []},
       'NEW': {'acc': [], 'f1': [], 'preds': [], 'trues': []}}
best_new_acc, best_new_state = 0.0, None

for fold, (tr, te) in enumerate(skf.split(range(len(dataset)), labels), 1):
    print(f"\n{'='*55}\nFOLD {fold}/5\n{'='*55}")
    Xtr = np.array([dataset[i].x.numpy().flatten() for i in tr])
    ytr = np.array([dataset[i].y.item() for i in tr])
    Xa, ya = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
    train_graphs = [Data(x=torch.tensor(xf, dtype=torch.float).view(-1, 1),
                         edge_index=base_ei, edge_attr=base_ea,
                         y=torch.tensor([yv], dtype=torch.long)) for xf, yv in zip(Xa, ya)]
    test_graphs = [dataset[i] for i in te]
    for kind in ('OLD', 'NEW'):
        t0 = time.time()
        acc, f1, state, preds, trues = train_eval(kind, train_graphs, test_graphs)
        res[kind]['acc'].append(acc); res[kind]['f1'].append(f1)
        res[kind]['preds'].extend(preds); res[kind]['trues'].extend(trues)
        print(f"  {kind}: acc={acc:.4f}  f1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if kind == 'NEW' and acc > best_new_acc:
            best_new_acc, best_new_state = acc, state

if best_new_state is not None:
    torch.save(best_new_state, CKPT_PATH)


# ════════════════════════════════════════════════════════════════════════════
# ОТЧЕТ
# ════════════════════════════════════════════════════════════════════════════
def summary(tag):
    a, f = res[tag]['acc'], res[tag]['f1']
    return f"{np.mean(a):.4f} ± {np.std(a):.4f}", f"{np.mean(f):.4f} ± {np.std(f):.4f}"

lines = []
lines.append("HEAD-TO-HEAD — OLD BioGNN vs NEW BioGNNv3 (196 гена, 5-Fold CV)")
lines.append("=" * 70)
lines.append(f"{'Модел':<28}{'Accuracy':<22}{'Macro F1'}")
lines.append("-" * 70)
oa, of = summary('OLD'); na, nf = summary('NEW')
lines.append(f"{'OLD BioGNN (3L, mean-pool)':<28}{oa:<22}{of}")
lines.append(f"{'NEW BioGNNv3 (JK+meanmax)':<28}{na:<22}{nf}")
lines.append("-" * 70)
lines.append(f"{'Референция: най-добър лог':<28}{'0.6861 ± 0.0201':<22}{'0.5429 ± 0.0611'}")
lines.append(f"{'Таван: RandomForest':<28}{'0.9072 ± 0.0197':<22}{'0.7896 ± 0.0493'}")
lines.append(f"{'Таван: LogReg (f1)':<28}{'0.8960 ± 0.0231':<22}{'0.8552 ± 0.0361'}")
lines.append("-" * 70)
for tag in ('OLD', 'NEW'):
    lines.append(f"\n=== {tag} — per-class (combined CV folds) ===")
    lines.append(classification_report(res[tag]['trues'], res[tag]['preds'],
                                        target_names=classes, digits=4, zero_division=0))
    lines.append(f"=== {tag} — confusion matrix (редове=истина, колони=предсказано) ===")
    cm = confusion_matrix(res[tag]['trues'], res[tag]['preds'])
    lines.append("           " + "  ".join(f"{c[5:]:>6}" for c in classes))
    for i, row in enumerate(cm):
        lines.append(f"{classes[i][5:]:>10} " + "  ".join(f"{v:>6}" for v in row))

report = "\n".join(lines)
print("\n" + report)
with open(OUT_PATH, 'w', encoding='utf-8') as fh:
    fh.write(report)
print(f"\nЗапазено: {OUT_PATH}\nNEW checkpoint: {CKPT_PATH} (best fold acc={best_new_acc:.4f})")
