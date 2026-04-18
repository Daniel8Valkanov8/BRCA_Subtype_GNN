"""
BRCA Subtype GNN - FastAPI Backend
"""
import io
import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy import stats
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

app = FastAPI(title="BRCA Subtype GNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Модел ──────────────────────────────────────────────────────────────────────

class BioGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, dropout=0.2)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=0.2)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False, dropout=0.2)
        self.lin1  = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2  = torch.nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch, return_attention=False):
        attn_weights = {}

        x, (ei1, aw1) = self.conv1(x, edge_index, return_attention_weights=True)
        attn_weights['layer1'] = (ei1.cpu(), aw1.cpu())
        x = F.relu(self.bn1(x))

        x, (ei2, aw2) = self.conv2(x, edge_index, return_attention_weights=True)
        attn_weights['layer2'] = (ei2.cpu(), aw2.cpu())
        x = F.relu(self.bn2(x))

        x, (ei3, aw3) = self.conv3(x, edge_index, return_attention_weights=True)
        attn_weights['layer3'] = (ei3.cpu(), aw3.cpu())
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)

        if return_attention:
            return x, attn_weights
        return x


# ── Зареждане при старт ────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_model_tcga_160genes.pt')
PPI_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'string_ppi_160genes.tsv')
EXPR_PATH  = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tcga_expression_174genes.csv')

CLASSES = ['BRCA_Basal', 'BRCA_Her2', 'BRCA_LumA', 'BRCA_LumB', 'BRCA_Normal']
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model     = None
gene_list = None
gene_map  = None
edge_index = None
ppi_df    = None
tcga_expr = None


def load_resources():
    global model, gene_list, gene_map, edge_index, ppi_df, tcga_expr

    ppi_df    = pd.read_csv(PPI_PATH, sep='\t')
    tcga_expr = pd.read_csv(EXPR_PATH, index_col=0)
    gene_list = list(tcga_expr.index)
    gene_map  = {g: i for i, g in enumerate(gene_list)}

    edges = []
    for _, row in ppi_df.iterrows():
        if row['node1'] in gene_map and row['node2'] in gene_map:
            i, j = gene_map[row['node1']], gene_map[row['node2']]
            edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)

    model = BioGNN(1, 128, len(CLASSES)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Модел зареден: {len(gene_list)} гена, {edge_index.shape[1]} ребра")


load_resources()


# ── Помощни функции ────────────────────────────────────────────────────────────

def build_patient_graph(expr_series: pd.Series):
    values = []
    for gene in gene_list:
        values.append(float(expr_series.get(gene, 0.0)))
    x = torch.tensor(values, dtype=torch.float).view(-1, 1).to(DEVICE)
    batch = torch.zeros(len(gene_list), dtype=torch.long).to(DEVICE)
    return x, batch


def top_attention_edges(attn_weights, top_n=50):
    ei, aw = attn_weights['layer3']
    scores = aw.mean(dim=1).detach().numpy()
    src, dst = ei[0].numpy(), ei[1].numpy()

    results = []
    seen = set()
    order = np.argsort(scores)[::-1]
    for idx in order:
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
            logits, attn = model(x, edge_index, batch, return_attention=True)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx  = int(np.argmax(probs))
        top_edges = top_attention_edges(attn)

        results.append({
            "patient_id":   patient_id,
            "prediction":   CLASSES[pred_idx],
            "confidence":   round(float(probs[pred_idx]) * 100, 1),
            "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)},
            "top_edges":    top_edges,
        })

    return JSONResponse(results)


@app.get("/discoveries")
def discoveries():
    """Агрегирани attention weights върху целия TCGA датасет."""
    edge_scores: dict[tuple, list] = {}

    patients = tcga_expr.columns[:200]  # sample за бързина
    for patient in patients:
        x, batch = build_patient_graph(tcga_expr[patient])
        with torch.no_grad():
            _, attn = model(x, edge_index, batch, return_attention=True)
        ei, aw = attn['layer3']
        scores = aw.mean(dim=1).detach().numpy()
        src, dst = ei[0].numpy(), ei[1].numpy()
        for s, d, sc in zip(src, dst, scores):
            pair = tuple(sorted([int(s), int(d)]))
            edge_scores.setdefault(pair, []).append(float(sc))

    top_edges = sorted(edge_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)[:80]

    nodes_set = set()
    edges_out = []
    for (s, d), sc_list in top_edges:
        g1, g2 = gene_list[s], gene_list[d]
        nodes_set.update([g1, g2])
        edges_out.append({
            "source": g1, "target": g2,
            "weight": round(float(np.mean(sc_list)), 6),
            "std":    round(float(np.std(sc_list)), 6),
        })

    nodes_out = [{"id": g} for g in nodes_set]
    return JSONResponse({"nodes": nodes_out, "edges": edges_out})


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
        "gene1": gene1,
        "gene2": gene2,
        "pearson_r":   round(float(pearson_r), 4),
        "pearson_p":   float(f"{pearson_p:.2e}"),
        "spearman_r":  round(float(spearman_r), 4),
        "spearman_p":  float(f"{spearman_p:.2e}"),
        "mean_expr1":  round(float(np.mean(expr1)), 4),
        "mean_expr2":  round(float(np.mean(expr2)), 4),
        "std_expr1":   round(float(np.std(expr1)), 4),
        "std_expr2":   round(float(np.std(expr2)), 4),
        "n_samples":   len(expr1),
        "significant": bool(pearson_p < 0.05),
    }
