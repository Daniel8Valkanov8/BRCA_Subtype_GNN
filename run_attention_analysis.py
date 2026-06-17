import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats
from torch_geometric.nn import GATConv

class BioGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(1, 128, heads=4, dropout=0.2)
        self.bn1   = torch.nn.BatchNorm1d(512)
        self.conv2 = GATConv(512, 128, heads=4, dropout=0.2)
        self.bn2   = torch.nn.BatchNorm1d(512)
        self.conv3 = GATConv(512, 128, heads=1, concat=False, dropout=0.2)
        self.lin1  = torch.nn.Linear(128, 64)
        self.lin2  = torch.nn.Linear(64, 5)

    def forward(self, x, edge_index):
        x, (ei1, aw1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = torch.nn.functional.relu(self.bn1(x))
        x, (ei2, aw2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = torch.nn.functional.relu(self.bn2(x))
        x, (ei3, aw3) = self.conv3(x, edge_index, return_attention_weights=True)
        return ei3, aw3

# Load model
model = BioGNN()
state = torch.load('best_model_tcga_160genes.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

# Load data
expr_df = pd.read_csv('data/tcga_expression_160genes.csv', index_col=0)
clin_df = pd.read_csv('data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
                      sep='\t', comment='#').dropna(subset=['Subtype'])
ppi_df  = pd.read_csv('data/string_ppi_160genes.tsv', sep='\t')

common  = sorted(set(clin_df['Sample ID']) & set(expr_df.columns))
expr_df = expr_df[common].fillna(0)
gene_list = list(expr_df.index)
gene_map  = {g: i for i, g in enumerate(gene_list)}

edges = []
for _, row in ppi_df.iterrows():
    if row['node1'] in gene_map and row['node2'] in gene_map:
        i, j = gene_map[row['node1']], gene_map[row['node2']]
        edges += [[i, j], [j, i]]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"Patients: {len(common)}, Genes: {len(gene_list)}, Edges: {edge_index.shape[1]}")

# Collect attention weights across ALL 981 patients
edge_scores = {}
for idx, pat in enumerate(common):
    vals = expr_df[pat].values.astype(float)
    mu, sd = vals.mean(), vals.std() + 1e-8
    x = torch.tensor((vals - mu) / sd, dtype=torch.float).view(-1, 1)
    with torch.no_grad():
        ei, aw = model(x, edge_index)
    scores = aw.mean(dim=1).numpy()
    src = ei[0].numpy()
    dst = ei[1].numpy()
    for s, d, sc in zip(src, dst, scores):
        pair = tuple(sorted([int(s), int(d)]))
        if pair[0] != pair[1]:
            edge_scores.setdefault(pair, []).append(float(sc))
    if idx % 200 == 0:
        print(f"  Processed {idx}/{len(common)} patients...")

print(f"Unique edges: {len(edge_scores)}")

# Top 100 edges by mean attention
top_edges = sorted(edge_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)[:100]

# Compute Pearson p-value for each
results = []
for (s, d), sc_list in top_edges:
    g1 = gene_list[s]
    g2 = gene_list[d]
    e1 = expr_df.loc[g1].values.astype(float)
    e2 = expr_df.loc[g2].values.astype(float)
    r, p = stats.pearsonr(e1, e2)
    results.append({
        'gene1': g1, 'gene2': g2,
        'mean_attention': float(np.mean(sc_list)),
        'std_attention':  float(np.std(sc_list)),
        'pearson_r': float(r),
        'pearson_p': float(p),
        'n_patients': len(common)
    })

df = pd.DataFrame(results)
df['rank'] = range(1, len(df) + 1)
df['significant_raw'] = df['pearson_p'] < 0.05
df['significant_bonf'] = df['pearson_p'] < (0.05 / 100)

# Save full results
df.to_csv('attention_analysis_results.csv', index=False)
print(f"\nResults saved to attention_analysis_results.csv")

sig_raw  = df[df['significant_raw']]
sig_bonf = df[df['significant_bonf']]
print(f"Top 100 edges by attention:")
print(f"  Significant p < 0.05:         {len(sig_raw)}")
print(f"  Significant Bonferroni p < 5e-4: {len(sig_bonf)}")
print()
print("=== TOP 30 EDGES (by attention, with statistics) ===")
print(f"{'#':>3} {'Gene1':<10} {'Gene2':<10} {'MeanAtt':>9} {'StdAtt':>8} {'r':>7} {'p':>12} {'Sig'}")
print("-" * 75)
for _, row in df.head(30).iterrows():
    bonf = "**" if row['significant_bonf'] else ("*" if row['significant_raw'] else "")
    print(f"{int(row['rank']):>3} {row['gene1']:<10} {row['gene2']:<10} "
          f"{row['mean_attention']:>9.6f} {row['std_attention']:>8.6f} "
          f"{row['pearson_r']:>7.4f} {row['pearson_p']:>12.3e} {bonf}")

print()
print("=== BONFERRONI-SIGNIFICANT PAIRS (p < 5e-4) ===")
print(f"{'#':>3} {'Gene1':<10} {'Gene2':<10} {'MeanAtt':>9} {'r':>7} {'p':>12}")
print("-" * 60)
for _, row in sig_bonf.sort_values('mean_attention', ascending=False).iterrows():
    print(f"{int(row['rank']):>3} {row['gene1']:<10} {row['gene2']:<10} "
          f"{row['mean_attention']:>9.6f} {row['pearson_r']:>7.4f} {row['pearson_p']:>12.3e}")

print()
print("=== ALL p < 0.05 SIGNIFICANT PAIRS ===")
print(f"{'#':>3} {'Gene1':<10} {'Gene2':<10} {'MeanAtt':>9} {'r':>7} {'p':>12}")
print("-" * 60)
for _, row in sig_raw.sort_values('mean_attention', ascending=False).iterrows():
    print(f"{int(row['rank']):>3} {row['gene1']:<10} {row['gene2']:<10} "
          f"{row['mean_attention']:>9.6f} {row['pearson_r']:>7.4f} {row['pearson_p']:>12.3e}")
