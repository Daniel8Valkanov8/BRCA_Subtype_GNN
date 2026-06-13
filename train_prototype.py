"""
ЛОКАЛЕН ПРОТОТИП — Head-to-head: OLD BioGNN vs NEW BioGNNv3
Същите 2 fold-а, идентичен pipeline (per-patient z-score + in-fold SMOTE +
class weights). Цел: бърза CPU валидация дали архитектурните корекции
подобряват над текущия GNN (5-fold acc=0.6861, f1=0.5429).

NB: 2 fold / 60 epochs / batch=32 — за бърз сигнал, НЕ финален резултат.
"""
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from src.data_loader import build_graph_dataset
from src.model import BioGNN
from src.model_v3 import BioGNNv3

HIDDEN, EPOCHS, BATCH, LR, WD, PATIENCE = 128, 60, 32, 5e-4, 1e-4, 12
N_FOLDS_RUN = 1   # бърз първи сигнал; разширяваме ако NEW > OLD
DEVICE = torch.device('cpu')

print("Зареждане на 196-генни данни...", flush=True)
clin = pd.read_csv('data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
                   sep='\t', comment='#').dropna(subset=['Subtype'])
expr = pd.read_csv('data/tcga_expression_198genes.csv', index_col=0)
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr = expr[common].fillna(0)
clin = clin.set_index('Sample ID').loc[common, ['Subtype']]
expr = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)
ppi = pd.read_csv('data/string_ppi_196genes.tsv', sep='\t')
dataset, encoder = build_graph_dataset(expr, ppi, clin)
NC = len(encoder.classes_)

base_ei, base_ea = dataset[0].edge_index, dataset[0].edge_attr
counts = clin['Subtype'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = (weights / weights.sum() * len(counts)).to(DEVICE)
labels = [g.y.item() for g in dataset]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_model(kind):
    if kind == 'OLD':
        return BioGNN(1, HIDDEN, NC, edge_dim=1).to(DEVICE)
    return BioGNNv3(1, HIDDEN, NC, edge_dim=1).to(DEVICE)


def train_eval(kind, train_graphs, test_graphs, n_params=None):
    tl = DataLoader(train_graphs, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(test_graphs, batch_size=BATCH)
    model = build_model(kind)
    np_count = sum(p.numel() for p in model.parameters())
    print(f"  [{kind}] параметри: {np_count:,} | старт на {EPOCHS} epochs...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    best_acc, best_f1, no_imp, last_ep = 0.0, 0.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        last_ep = epoch
        t_ep = time.time()
        model.train()
        tot = 0.0
        for b in tl:
            b = b.to(DEVICE)
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr)
            loss = crit(out, b.y)
            loss.backward()
            opt.step()
            tot += loss.item()
        sch.step(tot / max(len(tl), 1))
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(DEVICE)
                out = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr)
                ps.extend(out.argmax(1).cpu().numpy())
                ts.extend(b.y.cpu().numpy())
        acc = sum(p == t for p, t in zip(ps, ts)) / len(ts)
        f1 = f1_score(ts, ps, average='macro', zero_division=0)
        if acc > best_acc:
            best_acc, best_f1, no_imp = acc, f1, 0
        else:
            no_imp += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{kind}] epoch {epoch:3d}/{EPOCHS} | loss={tot/max(len(tl),1):.3f} "
                  f"| val_acc={acc:.4f} | val_f1={f1:.4f} | best_acc={best_acc:.4f} "
                  f"| {time.time()-t_ep:.1f}s/ep", flush=True)
        if no_imp >= PATIENCE:
            print(f"  [{kind}] early stop @ epoch {epoch}", flush=True)
            break
    return best_acc, best_f1, last_ep


results = {'OLD': [], 'NEW': []}
for fold, (tr, te) in enumerate(skf.split(range(len(dataset)), labels), 1):
    if fold > N_FOLDS_RUN:
        break
    Xtr = np.array([dataset[i].x.numpy().flatten() for i in tr])
    ytr = np.array([dataset[i].y.item() for i in tr])
    Xa, ya = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
    train_graphs = [Data(x=torch.tensor(xf, dtype=torch.float).view(-1, 1),
                         edge_index=base_ei, edge_attr=base_ea,
                         y=torch.tensor([yv], dtype=torch.long))
                    for xf, yv in zip(Xa, ya)]
    test_graphs = [dataset[i] for i in te]
    for kind in ('OLD', 'NEW'):
        t0 = time.time()
        acc, f1, ep = train_eval(kind, train_graphs, test_graphs)
        results[kind].append((acc, f1))
        print(f"Fold {fold} | {kind:3s} -> acc={acc:.4f}  f1={f1:.4f}  "
              f"(stop@{ep}, {time.time()-t0:.0f}s)", flush=True)

print("\n=== ОБОБЩЕНИЕ (средно по 2 fold) ===", flush=True)
for k in ('OLD', 'NEW'):
    a = np.mean([x[0] for x in results[k]])
    f = np.mean([x[1] for x in results[k]])
    print(f"  {k:3s}: acc={a:.4f}  macro-f1={f:.4f}", flush=True)
print(f"\n  Референция (5-fold пълен): OLD 196г = acc 0.6861 / f1 0.5429", flush=True)
print(f"  Baseline таван:            RF 0.9072 / LogReg f1 0.8552", flush=True)
