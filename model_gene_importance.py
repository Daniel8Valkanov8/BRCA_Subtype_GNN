# -*- coding: utf-8 -*-
"""
Per-class градиентна значимост (saliency) на гените според МОДЕЛА
best_model_tcga_196genes_09.06.pt.

Метод: за всеки пациент правим forward pass и backward на логита на всеки клас k
спрямо входните node-характеристики x (генна експресия). |dlogit_k/dx_g| показва
колко чувствителна е прогнозата за субтип k към експресията на ген g.

За класова значимост осредняваме saliency по пациентите с ИСТИНСКИ етикет k
(„за пациенти от субтип k, кои гени най-силно влияят на разпознаването му").

Изход: gene_importance_by_class.csv  (196 гена x 5 класа + назначен субтип).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
from model import BioGNN
from data_loader import build_graph_dataset

DEVICE = 'cpu'
MODEL_PATH = 'best_model_tcga_196genes_09.06.pt'

# --- данни (същата подготовка като verify_model_genes.py) -------------------
clin = pd.read_csv('data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
                   sep='\t', comment='#').dropna(subset=['Subtype'])
expr = pd.read_csv('data/tcga_expression_198genes.csv', index_col=0)
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr = expr[common].fillna(0)
clin = clin.set_index('Sample ID').loc[common, ['Subtype']]
expr = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)  # per-patient z
ppi = pd.read_csv('data/string_ppi_196genes.tsv', sep='\t')

genes = list(expr.index)
ds, le = build_graph_dataset(expr, ppi, clin)
classes = list(le.classes_)
print(f"Класове: {classes}")
print(f"Гени: {len(genes)} | Пациенти: {len(ds)}")

# --- модел ------------------------------------------------------------------
model = BioGNN(1, 128, 5, edge_dim=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()  # BatchNorm -> running stats, dropout off

n_genes, n_cls = len(genes), len(classes)
# натрупваме saliency и брой за всеки (клас по истински етикет)
sal_sum = np.zeros((n_genes, n_cls))   # sal_sum[:,k] = сума |grad logit_k| по пациенти с y=k
cls_count = np.zeros(n_cls)

batch0 = torch.zeros(n_genes, dtype=torch.long, device=DEVICE)  # един граф

for idx, data in enumerate(ds):
    x = data.x.clone().detach().to(DEVICE).requires_grad_(True)
    ei = data.edge_index.to(DEVICE)
    ea = data.edge_attr.to(DEVICE)
    y = int(data.y.item())

    out = model(x, ei, batch0, ea)          # [1, 5]
    k = y                                    # логита на истинския клас
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    out[0, k].backward()
    g = x.grad.detach().abs().view(-1).cpu().numpy()   # [196]
    sal_sum[:, k] += g
    cls_count[k] += 1

    if (idx + 1) % 200 == 0:
        print(f"  обработени {idx+1}/{len(ds)} пациента")

# средна saliency по клас
importance = sal_sum / np.clip(cls_count, 1, None)   # [196, 5]

imp_df = pd.DataFrame(importance, index=genes, columns=classes)

# нормализация по клас (z-score) за да са класовете сравними при argmax
z = (imp_df - imp_df.mean(axis=0)) / (imp_df.std(axis=0) + 1e-9)
imp_df['assigned_subtype'] = z.idxmax(axis=1)         # клас с най-силна (релативна) значимост
imp_df['assigned_score'] = z.max(axis=1)

imp_df.index.name = 'gene'
OUT = 'gene_importance_by_class.csv'
imp_df.to_csv(OUT, encoding='utf-8')
print(f"\n[OK] Записан: {OUT}")
print("\nРазпределение на назначените субтипове (по модела):")
print(imp_df['assigned_subtype'].value_counts().to_string())
print("\nТоп 8 гена по клас (по нормализирана saliency):")
for c in classes:
    top = z[c].sort_values(ascending=False).head(8).index.tolist()
    print(f"  {c:14s}: {', '.join(top)}")
