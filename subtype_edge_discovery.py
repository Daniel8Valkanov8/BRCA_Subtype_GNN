# -*- coding: utf-8 -*-
"""
ЯДРО: Откриване на субтип-разграничаващи връзки ген-ген (ребра) от GNN.

Модел: best_model_tcga_196genes_09.06.pt  (BioGNN, GATv2Conv, 196 гена)

ДВОЕН ФИЛТЪР (всяко ребро трябва да мине И двата теста):
  Тест A — attention хетерогенност (ЗАВИСИ от модела):
      patient-level attention на реброто, групиран по 5 субтипа.
      Kruskal-Wallis: различава ли се attention между субтиповете? -> p_A
  Тест B — диференциална ко-експресия (НЕЗАВИСИМ от модела):
      Pearson корелация на двата гена ВЪТРЕ във всеки субтип (глобални z-scores).
      Тест за равенство на k корелации (Fisher-z, chi^2): различава ли се
      връзката между субтиповете? -> p_B
  И двата p минават Benjamini-Hochberg FDR. Запазваме edges с двата FDR < 0.05.

Изход:
  subtype_edge_discovery.csv  — всички ребра + per-subtype attention/корелация + p/FDR
  subtype_edge_discovery_report.txt — обобщение и топ кандидати
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
from scipy import stats
from model import BioGNN

DEVICE = 'cpu'
MODEL_PATH = 'best_model_tcga_196genes_09.06.pt'
ATT_LAYER = 'layer3'          # финален aggregation слой
FDR_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Данни
# ---------------------------------------------------------------------------
clin = pd.read_csv('data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
                   sep='\t', comment='#').dropna(subset=['Subtype'])
expr = pd.read_csv('data/tcga_expression_198genes.csv', index_col=0)   # ГЛОБАЛНИ z-scores
common = sorted(set(clin['Sample ID']) & set(expr.columns))
expr = expr[common].fillna(0)
subtype = clin.set_index('Sample ID').loc[common, 'Subtype']
ppi = pd.read_csv('data/string_ppi_196genes.tsv', sep='\t')

genes = list(expr.index)
gene_map = {g: i for i, g in enumerate(genes)}
subtypes = sorted(subtype.unique())
print(f"Пациенти: {len(common)} | Гени: {len(genes)} | Субтипове: {subtypes}")

# уникални неориентирани ребра (в реда на PPI), + combined_score
edge_pairs = []          # (i, j) i<j
edge_score = []
seen = set()
for _, row in ppi.iterrows():
    if row['node1'] in gene_map and row['node2'] in gene_map:
        i, j = sorted((gene_map[row['node1']], gene_map[row['node2']]))
        if (i, j) not in seen:
            seen.add((i, j))
            edge_pairs.append((i, j))
            edge_score.append(float(row['combined_score']))
n_edges = len(edge_pairs)
pair_to_eidx = {p: k for k, p in enumerate(edge_pairs)}
print(f"Уникални ребра: {n_edges}")

# ориентиран edge_index/edge_attr за модела (както в data_loader)
ei_list, ea_list = [], []
for (i, j), sc in zip(edge_pairs, edge_score):
    s = sc * 1000.0 if max(edge_score) <= 0.01 else sc
    ei_list += [[i, j], [j, i]]
    ea_list += [[s], [s]]
edge_index = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(ea_list, dtype=torch.float)

# ---------------------------------------------------------------------------
# Модел
# ---------------------------------------------------------------------------
model = BioGNN(1, 128, 5, edge_dim=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# ---------------------------------------------------------------------------
# ЕТАП 1 — patient-level attention матрица [пациенти × ребра]
# ---------------------------------------------------------------------------
att_matrix = np.full((len(common), n_edges), np.nan, dtype=np.float32)
batch0 = torch.zeros(len(genes), dtype=torch.long, device=DEVICE)

for pidx, pat in enumerate(common):
    vals = expr[pat].values.astype(float)
    mu, sd = vals.mean(), vals.std() + 1e-8
    x = torch.tensor((vals - mu) / sd, dtype=torch.float).view(-1, 1).to(DEVICE)  # per-patient z (вход на модела)
    with torch.no_grad():
        _, attn = model(x, edge_index, batch0, edge_attr=edge_attr, return_attention=True)
    ei, aw = attn[ATT_LAYER]
    sc = aw.mean(dim=1).cpu().numpy()
    src = ei[0].cpu().numpy(); dst = ei[1].cpu().numpy()
    acc = np.zeros(n_edges); cnt = np.zeros(n_edges)
    for s, d, v in zip(src, dst, sc):
        if s == d:
            continue                       # self-loop
        p = (s, d) if s < d else (d, s)
        k = pair_to_eidx.get(p)
        if k is not None:
            acc[k] += v; cnt[k] += 1
    mask = cnt > 0
    att_matrix[pidx, mask] = acc[mask] / cnt[mask]   # средно по двете посоки
    if (pidx + 1) % 200 == 0:
        print(f"  attention: {pidx+1}/{len(common)}")

# subtype индекси на пациентите
sub_arr = subtype.values
group_idx = {st: np.where(sub_arr == st)[0] for st in subtypes}

# ---------------------------------------------------------------------------
# ЕТАП 2 — двата теста за всяко ребро
# ---------------------------------------------------------------------------
expr_vals = expr.values.astype(float)   # [гени × пациенти] глобални z-scores

def equal_correlations_p(r_list, n_list):
    """Fisher-z chi^2 тест за равенство на k Pearson корелации."""
    r_list = np.clip(np.array(r_list), -0.999, 0.999)
    n_list = np.array(n_list)
    valid = n_list > 3
    if valid.sum() < 2:
        return np.nan
    z = np.arctanh(r_list[valid])
    w = n_list[valid] - 3
    zbar = np.sum(w * z) / np.sum(w)
    chi2 = np.sum(w * (z - zbar) ** 2)
    df = valid.sum() - 1
    return float(stats.chi2.sf(chi2, df))

rows = []
for k, (i, j) in enumerate(edge_pairs):
    att_col = att_matrix[:, k]

    # --- Тест A: Kruskal-Wallis на attention по субтип ---
    groups_att = [att_col[group_idx[st]] for st in subtypes]
    groups_att = [g[~np.isnan(g)] for g in groups_att]
    try:
        if all(len(g) > 0 for g in groups_att) and any(np.ptp(g) > 0 for g in groups_att):
            _, pA = stats.kruskal(*groups_att)
        else:
            pA = np.nan
    except Exception:
        pA = np.nan

    # per-subtype средно attention
    att_means = {st: float(np.nanmean(att_col[group_idx[st]])) for st in subtypes}
    top_att_subtype = max(att_means, key=att_means.get)

    # --- Тест B: диференциална ко-експресия (глобални z-scores) ---
    ei_expr = expr_vals[i]; ej_expr = expr_vals[j]
    r_list, n_list, r_by_sub = [], [], {}
    for st in subtypes:
        gi = group_idx[st]
        if len(gi) > 3:
            r, _ = stats.pearsonr(ei_expr[gi], ej_expr[gi])
        else:
            r = np.nan
        r_by_sub[st] = r
        r_list.append(r if not np.isnan(r) else 0.0)
        n_list.append(len(gi))
    pB = equal_correlations_p(r_list, n_list)

    rows.append({
        'gene1': genes[i], 'gene2': genes[j],
        'string_score': edge_score[k],
        'mean_attention': float(np.nanmean(att_col)),
        **{f'att_{st}': att_means[st] for st in subtypes},
        'top_att_subtype': top_att_subtype,
        'attention_kruskal_p': pA,
        **{f'r_{st}': r_by_sub[st] for st in subtypes},
        'diffcoexp_p': pB,
    })

df = pd.DataFrame(rows)

# --- BH-FDR върху двата p (само върху валидните) ---
def bh_fdr(pvals):
    p = np.array(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    order = np.argsort(p[ok])
    ranked = p[ok][order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty(m); tmp[order] = np.clip(q, 0, 1)
    out[ok] = tmp
    return out

df['attention_fdr'] = bh_fdr(df['attention_kruskal_p'])
df['diffcoexp_fdr'] = bh_fdr(df['diffcoexp_p'])
df['sig_attention'] = df['attention_fdr'] < FDR_ALPHA
df['sig_diffcoexp'] = df['diffcoexp_fdr'] < FDR_ALPHA
df['sig_BOTH'] = df['sig_attention'] & df['sig_diffcoexp']

# ефект-размер за класиране: разлика max-min per-subtype корелация (обръщане на връзката)
rcols = [f'r_{st}' for st in subtypes]
df['corr_range'] = df[rcols].max(axis=1) - df[rcols].min(axis=1)
df['att_range'] = df[[f'att_{st}' for st in subtypes]].max(axis=1) - df[[f'att_{st}' for st in subtypes]].min(axis=1)
df = df.sort_values(['sig_BOTH', 'corr_range'], ascending=[False, False]).reset_index(drop=True)

OUT_CSV = 'subtype_edge_discovery.csv'
df.to_csv(OUT_CSV, index=False, encoding='utf-8')

# ---------------------------------------------------------------------------
# Доклад
# ---------------------------------------------------------------------------
n_A = int(df['sig_attention'].sum())
n_B = int(df['sig_diffcoexp'].sum())
n_both = int(df['sig_BOTH'].sum())
lines = []
lines.append("=" * 78)
lines.append("ОТКРИВАНЕ НА СУБТИП-РАЗГРАНИЧАВАЩИ ВРЪЗКИ ГЕН-ГЕН (двоен филтър)")
lines.append(f"Модел: {MODEL_PATH} | слой: {ATT_LAYER}")
lines.append("=" * 78)
lines.append(f"Общо ребра: {n_edges}")
lines.append(f"Тест A (attention хетерогенност) значими (FDR<{FDR_ALPHA}): {n_A}")
lines.append(f"Тест B (диференциална ко-експресия) значими (FDR<{FDR_ALPHA}): {n_B}")
lines.append(f">>> И ДВАТА (кандидати за анализ): {n_both}")
lines.append("")
lines.append("ТОП 25 кандидати (минали двойния филтър, по обръщане на ко-експресията):")
lines.append("-" * 78)
hdr = f"{'#':>3} {'Gene1':<9}{'Gene2':<9} {'top_sub':<12} {'attP':>9} {'dcP':>9} {'corrRng':>7}"
lines.append(hdr)
top = df[df['sig_BOTH']].head(25)
for n, (_, row) in enumerate(top.iterrows(), 1):
    lines.append(f"{n:>3} {row['gene1']:<9}{row['gene2']:<9} {row['top_att_subtype']:<12} "
                 f"{row['attention_kruskal_p']:>9.2e} {row['diffcoexp_p']:>9.2e} {row['corr_range']:>7.2f}")
lines.append("")
lines.append("СЛЕДВАЩА СТЪПКА (Етап 3): литературен филтър (PubMed co-citation) върху")
lines.append(f"тези {n_both} ребра -> разделяне на 'нови' (Цел 1) vs 'потвърдени' (Цел 2).")
report = "\n".join(lines)
with open('subtype_edge_discovery_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n" + report)
print(f"\n[OK] {OUT_CSV} | subtype_edge_discovery_report.txt")
