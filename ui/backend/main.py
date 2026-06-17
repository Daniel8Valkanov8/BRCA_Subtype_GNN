"""
BRCA Subtype GNN - FastAPI Backend
Model: GATv2Conv + edge features (STRING combined_score), Acc=0.6136
"""
import io
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from scipy import stats
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torch_geometric.data import Data
import torch.nn.functional as F

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model import BioGNN

app = FastAPI(title="BRCA Subtype GNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Зареждане при старт ────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_model_tcga_196genes_09.06.pt')
PPI_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'string_ppi_196genes.tsv')
EXPR_PATH  = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tcga_expression_198genes.csv')

CLASSES = ['BRCA_Basal', 'BRCA_Her2', 'BRCA_LumA', 'BRCA_LumB', 'BRCA_Normal']
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model             = None
gene_list         = None
gene_map          = None
edge_index        = None
edge_attr         = None
ppi_df            = None
tcga_expr         = None
brain_global_cache = None


def load_resources():
    global model, gene_list, gene_map, edge_index, edge_attr, ppi_df, tcga_expr

    ppi_df    = pd.read_csv(PPI_PATH, sep='\t')
    tcga_expr = pd.read_csv(EXPR_PATH, index_col=0)
    gene_list = list(tcga_expr.index)
    gene_map  = {g: i for i, g in enumerate(gene_list)}

    edges      = []
    edge_scores = []
    has_score  = 'combined_score' in ppi_df.columns
    for _, row in ppi_df.iterrows():
        if row['node1'] in gene_map and row['node2'] in gene_map:
            i, j  = gene_map[row['node1']], gene_map[row['node2']]
            score = float(row['combined_score']) if has_score else 1.0
            edges      += [[i, j], [j, i]]
            edge_scores += [[score], [score]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)
    edge_attr  = torch.tensor(edge_scores, dtype=torch.float).to(DEVICE)

    model = BioGNN(1, 128, len(CLASSES), edge_dim=1).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {len(gene_list)} genes, {edge_index.shape[1]} edges, "
          f"edge_attr [{edge_attr.min().item():.3f}, {edge_attr.max().item():.3f}]")


load_resources()


# ── Помощни функции ────────────────────────────────────────────────────────────

def extract_layer_brain_data(attn_weights: dict, top_n: int = 80) -> dict:
    """Per-layer attention: top edges + per-node incoming attention sum."""
    result = {}
    for lname in ['layer1', 'layer2', 'layer3']:
        ei, aw = attn_weights[lname]
        scores   = aw.mean(dim=1).detach().numpy()
        src_arr  = ei[0].numpy()
        dst_arr  = ei[1].numpy()

        node_attn = np.zeros(len(gene_list))
        for d, sc in zip(dst_arr, scores):
            node_attn[int(d)] += float(sc)

        seen, edges = set(), []
        for idx in np.argsort(scores)[::-1]:
            s, d, sc = int(src_arr[idx]), int(dst_arr[idx]), float(scores[idx])
            pair = tuple(sorted([s, d]))
            if pair in seen or s == d:
                continue
            seen.add(pair)
            edges.append({"source": gene_list[s], "target": gene_list[d], "weight": round(sc, 6)})
            if len(edges) >= top_n:
                break

        result[lname] = {
            "edges": edges,
            "node_attention": {gene_list[i]: round(float(v), 6) for i, v in enumerate(node_attn)},
        }
    return result


def build_patient_graph(expr_series: pd.Series):
    values = [float(expr_series.get(gene, 0.0)) for gene in gene_list]
    x     = torch.tensor(values, dtype=torch.float).view(-1, 1).to(DEVICE)
    mu, sd = x.mean(), x.std() + 1e-8
    x     = (x - mu) / sd
    batch = torch.zeros(len(gene_list), dtype=torch.long).to(DEVICE)
    return x, batch


def top_attention_edges(attn_weights, top_n=50):
    ei, aw = attn_weights['layer3']
    scores = aw.mean(dim=1).detach().numpy()
    src, dst = ei[0].numpy(), ei[1].numpy()

    results = []
    seen    = set()
    for idx in np.argsort(scores)[::-1]:
        s, d, sc = int(src[idx]), int(dst[idx]), float(scores[idx])
        pair = tuple(sorted([s, d]))
        if pair in seen:
            continue
        seen.add(pair)
        results.append({
            "source": gene_list[s],
            "target": gene_list[d],
            "weight": round(sc, 6)
        })
        if len(results) >= top_n:
            break
    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "genes": len(gene_list), "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), index_col=0)
    except Exception as e:
        raise HTTPException(400, f"Грешка при четене на файла: {e}")

    results = []
    for patient_id in df.columns:
        x, batch = build_patient_graph(df[patient_id])
        with torch.no_grad():
            logits, attn = model(x, edge_index, batch,
                                 edge_attr=edge_attr, return_attention=True)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx    = int(np.argmax(probs))
        top_edges   = top_attention_edges(attn)
        brain_layers = extract_layer_brain_data(attn)

        results.append({
            "patient_id":    patient_id,
            "prediction":    CLASSES[pred_idx],
            "confidence":    round(float(probs[pred_idx]) * 100, 1),
            "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)},
            "top_edges":     top_edges,
            "brain_layers":  brain_layers,
        })

    return JSONResponse(results)


@app.get("/discoveries")
def discoveries():
    """Агрегирани attention weights върху целия TCGA датасет."""
    edge_scores: dict[tuple, list] = {}

    for patient in tcga_expr.columns:
        x, batch = build_patient_graph(tcga_expr[patient])
        with torch.no_grad():
            _, attn = model(x, edge_index, batch,
                            edge_attr=edge_attr, return_attention=True)
        ei, aw = attn['layer3']
        scores = aw.mean(dim=1).detach().numpy()
        src, dst = ei[0].numpy(), ei[1].numpy()
        for s, d, sc in zip(src, dst, scores):
            pair = tuple(sorted([int(s), int(d)]))
            edge_scores.setdefault(pair, []).append(float(sc))

    top_edges = sorted(edge_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)[:80]
    top_edges = [(pair, sc) for pair, sc in top_edges if pair[0] != pair[1]]

    nodes_set = set()
    edges_out = []
    for (s, d), sc_list in top_edges:
        g1, g2 = gene_list[s], gene_list[d]
        nodes_set.update([g1, g2])
        edges_out.append({
            "source": g1,
            "target": g2,
            "weight": round(float(np.mean(sc_list)), 6),
            "std":    round(float(np.std(sc_list)), 6),
        })

    nodes_out = [{"id": g} for g in nodes_set]
    return JSONResponse({"nodes": nodes_out, "edges": edges_out})


@app.get("/brain")
def brain():
    """3D attention flow — gene positions + per-layer aggregated attention over all patients."""
    global brain_global_cache
    if brain_global_cache is not None:
        return JSONResponse(brain_global_cache)

    n_genes = len(gene_list)

    # ── Gene 2-D layout ────────────────────────────────────────────────────────
    if _HAS_NX:
        G = nx.Graph()
        G.add_nodes_from(range(n_genes))
        ei_cpu = edge_index.cpu()
        ea_cpu = edge_attr[:, 0].cpu()
        for k in range(ei_cpu.shape[1]):
            s, d = int(ei_cpu[0, k]), int(ei_cpu[1, k])
            if s < d:
                G.add_edge(s, d, weight=float(ea_cpu[k]))
        pos = nx.spring_layout(G, k=1.5 / math.sqrt(n_genes), iterations=80, seed=42)
        pos_xy = {i: (float(pos[i][0]), float(pos[i][1])) for i in range(n_genes)}
    else:
        pos_xy = {
            i: (math.cos(2 * math.pi * i / n_genes), math.sin(2 * math.pi * i / n_genes))
            for i in range(n_genes)
        }

    # ── Aggregate attention over all patients ──────────────────────────────────
    edge_sums = {l: {} for l in ['layer1', 'layer2', 'layer3']}
    node_sums = {l: np.zeros(n_genes) for l in ['layer1', 'layer2', 'layer3']}
    n_patients = 0

    for patient in tcga_expr.columns:
        x, batch = build_patient_graph(tcga_expr[patient])
        with torch.no_grad():
            _, attn = model(x, edge_index, batch, edge_attr=edge_attr, return_attention=True)

        for lname in ['layer1', 'layer2', 'layer3']:
            ei, aw = attn[lname]
            scores = aw.mean(dim=1).detach().numpy()
            src_a, dst_a = ei[0].numpy(), ei[1].numpy()
            for s, d, sc in zip(src_a, dst_a, scores):
                pair = tuple(sorted([int(s), int(d)]))
                edge_sums[lname][pair] = edge_sums[lname].get(pair, 0.0) + float(sc)
                node_sums[lname][int(d)] += float(sc)
        n_patients += 1

    n = max(n_patients, 1)

    layers_out = {}
    for lname in ['layer1', 'layer2', 'layer3']:
        top_pairs = sorted(edge_sums[lname].items(), key=lambda kv: kv[1], reverse=True)[:120]
        edges_out = [
            {"source": gene_list[s], "target": gene_list[d], "weight": round(total / n, 6)}
            for (s, d), total in top_pairs if s != d
        ]
        node_attn_out = {
            gene_list[i]: round(float(v / n), 6) for i, v in enumerate(node_sums[lname])
        }
        layers_out[lname] = {"edges": edges_out, "node_attention": node_attn_out}

    genes_out = [
        {"id": gene_list[i], "x": pos_xy[i][0], "y": pos_xy[i][1]}
        for i in range(n_genes)
    ]

    brain_global_cache = {"genes": genes_out, "layers": layers_out}
    return JSONResponse(brain_global_cache)


@app.get("/statistics/{gene1}/{gene2}")
def gene_statistics(gene1: str, gene2: str):
    """Статистики за двойка гени от TCGA експресионните данни."""
    if gene1 not in gene_map or gene2 not in gene_map:
        raise HTTPException(404, "Генът не е намерен в датасета")

    expr1 = tcga_expr.loc[gene1].values.astype(float)
    expr2 = tcga_expr.loc[gene2].values.astype(float)

    pearson_r,  pearson_p  = stats.pearsonr(expr1, expr2)
    spearman_r, spearman_p = stats.spearmanr(expr1, expr2)

    return {
        "gene1":      gene1,
        "gene2":      gene2,
        "pearson_r":  round(float(pearson_r), 4),
        "pearson_p":  float(f"{pearson_p:.2e}"),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": float(f"{spearman_p:.2e}"),
        "mean_expr1": round(float(np.mean(expr1)), 4),
        "mean_expr2": round(float(np.mean(expr2)), 4),
        "std_expr1":  round(float(np.std(expr1)), 4),
        "std_expr2":  round(float(np.std(expr2)), 4),
        "n_samples":  len(expr1),
        "significant": bool(pearson_p < 0.05),
    }
