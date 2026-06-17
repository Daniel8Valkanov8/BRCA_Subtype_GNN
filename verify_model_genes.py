"""
Проверка: на колко гена е трениран всеки модел.
Пуска двата модела срещу двата датасета (160 и 196 гена) и сравнява
точността спрямо истинските подтипове. Тренировъчният панел дава висока
точност; несъвместимият панел се срива.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model import BioGNN
from data_loader import build_graph_dataset

DEVICE = 'cpu'

def make_dataset(expr_csv, ppi_tsv):
    clin = pd.read_csv('data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
                       sep='\t', comment='#').dropna(subset=['Subtype'])
    expr = pd.read_csv(expr_csv, index_col=0)
    common = sorted(set(clin['Sample ID']) & set(expr.columns))
    expr = expr[common].fillna(0)
    clin = clin.set_index('Sample ID').loc[common, ['Subtype']]
    expr = expr.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)
    ppi = pd.read_csv(ppi_tsv, sep='\t')
    ds, le = build_graph_dataset(expr, ppi, clin)
    return ds, expr.shape[0]

def evaluate(model_path, dataset):
    model = BioGNN(1, 128, 5, edge_dim=1).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(dataset, batch_size=32)
    preds, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            out = model(b.x, b.edge_index, b.batch, b.edge_attr)
            preds += out.argmax(1).tolist()
            ys += b.y.tolist()
    return accuracy_score(ys, preds), f1_score(ys, preds, average='macro')

print("Строя датасетите...\n")
ds160, n160 = make_dataset('data/tcga_expression_160genes.csv', 'data/string_ppi_160genes.tsv')
ds196, n196 = make_dataset('data/tcga_expression_198genes.csv', 'data/string_ppi_196genes.tsv')
print(f"\n160-genes dataset: {len(ds160)} pts, {n160} genes/nodes")
print(f"196-genes dataset: {len(ds196)} pts, {n196} genes/nodes\n")

models = {
    'best_model_tcga_160genes_06,08,26.pt  (8 юни, документиран)': 'best_model_tcga_160genes_06,08,26.pt',
    'best_model_tcga_196genes_09.06.pt      (9 юни, 196 гена)': 'best_model_tcga_196genes_09.06.pt',
}

print(f"{'МОДЕЛ':<58} {'на 160-панел':>16} {'на 196-панел':>16}")
print("-" * 92)
for label, path in models.items():
    a160, f160 = evaluate(path, ds160)
    a196, f196 = evaluate(path, ds196)
    print(f"{label:<58} Acc={a160:.3f} F1={f160:.3f}   Acc={a196:.3f} F1={f196:.3f}")
