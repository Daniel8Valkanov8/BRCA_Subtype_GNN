# -*- coding: utf-8 -*-
"""
generate_model_versions_report.py
=================================
Произвежда:
  - model_versions_summary.csv   — структурирана таблица на всички версии
  - model_versions_report.pdf    — сравнителен PDF с диаграми A..H (на български)

Числата са извлечени от текстовите result-файлове КАКТО СА, а архитектурите
са ВЕРИФИЦИРАНИ чрез презареждане на .pt чекпойнтите (inspect на state_dict).
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "DejaVu Sans"   # поддържа кирилица
plt.rcParams["axes.unicode_minus"] = False

GEN_DATE = datetime.date.today().isoformat()

# ============================================================================
# 1. ДАННИ ЗА ВЕРСИИТЕ (verified_from_checkpoint, където е приложимо)
# ============================================================================
# Полета: version_id, label, date, script, data_source, n_patients, n_genes,
#         layer_type, n_layers, hidden, heads, node_feat, edge_used, jk, dual_readout,
#         batchnorm, params, loss, optimizer, scheduler, smote, dropedge,
#         acc, acc_std, f1, f1_std, checkpoint, verified, metrics_source, whats_new
VERSIONS = [
    dict(version_id="V1", label="V1 · 27 гена\nGATConv 2L", date="2026-03",
         script="main.ipynb / train_prototype.py", data_source="TCGA", n_patients="~880",
         n_genes=27, layer_type="GATConv", n_layers=2, hidden=64, heads="?",
         node_feat=1, edge_used="не", jk="не", dual_readout="не", batchnorm="?",
         params=None, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="не", dropedge="не",
         acc=0.5658, acc_std=None, f1=0.4591, f1_std=None,
         checkpoint="(няма запазен)", verified=False,
         metrics_source="Материали §6.2 / §7", verified_ckpt=False, tag="прототип 27г",
         whats_new="Базов прототип: малък 27-генен панел, проста PPI мрежа (312 ребра). Без нормализация, без SMOTE, без early stopping."),

    dict(version_id="V3", label="V3 · 160 гена\n+METABRIC", date="2026-04",
         script="train_expanded.py", data_source="TCGA+METABRIC", n_patients="~2737",
         n_genes=160, layer_type="GATConv", n_layers=3, hidden=128, heads=4,
         node_feat=1, edge_used="не", jk="не", dual_readout="не", batchnorm="да",
         params=None, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="?", dropedge="не",
         acc=0.3745, acc_std=None, f1=0.3562, f1_std=None,
         checkpoint="(няма запазен)", verified=False,
         metrics_source="Материали §6.2 / §7", verified_ckpt=False,
         tag="+METABRIC (регрес.)",
         whats_new="Опит за добавяне на METABRIC за повече данни. РЕГРЕСИЯ заради batch effects между платформите — METABRIC изоставен."),

    dict(version_id="V4", label="V4 · 160 гена\nGATConv 3L", date="2026-04-18",
         script="train_tcga_only.py", data_source="TCGA", n_patients=981,
         n_genes=160, layer_type="GATConv", n_layers=3, hidden=128, heads=4,
         node_feat=1, edge_used="не", jk="не", dual_readout="не (mean pool)", batchnorm="да (bn1,bn2)",
         params=344327, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="да (in-fold)", dropedge="не",
         acc=0.5851, acc_std=None, f1=0.4353, f1_std=None,
         checkpoint="best_model_tcga_160genes.pt", verified=True,
         metrics_source="Материали §6.2 / v4_results.txt", verified_ckpt=True,
         tag="TCGA-only · норм. · SMOTE",
         whats_new="Само TCGA + per-patient z-score норм. + class weights + SMOTE (in-fold) + early stopping. Най-голямото подобрение дойде от данните, не от архитектурата."),

    dict(version_id="V5", label="V5 · 160 гена\nGATv2 (buggy edge)", date="2026-05-04",
         script="train (Colab T4)", data_source="TCGA", n_patients=981,
         n_genes=160, layer_type="GATv2Conv", n_layers=3, hidden=128, heads=4,
         node_feat=1, edge_used="да, но ~0 (бъг)", jk="не", dual_readout="не (mean pool)", batchnorm="да",
         params=674823, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="да (in-fold)", dropedge="не",
         acc=0.5535, acc_std=0.0441, f1=0.4001, f1_std=0.0290,
         checkpoint="Last_model_tcga_160genes(1).pt", verified=True,
         metrics_source="baseline_results.txt / Материали §10.2", verified_ckpt=True,
         tag="GATv2 +edge (бъг)",
         whats_new="Преход към GATv2Conv (динамично attention) + edge_dim=1. НО двойна нормализация прави edge_attr≈1e-7 → реален ребрен сигнал липсва; затова РЕГРЕСИЯ спрямо V4."),

    dict(version_id="ExpA", label="Exp A · 160 гена\nGATConv (repro)", date="2026-05-12",
         script="train_colab_headtohead.py", data_source="TCGA", n_patients=981,
         n_genes="160 (157 ефект.)", layer_type="GATConv", n_layers=3, hidden=128, heads=4,
         node_feat=1, edge_used="не", jk="не", dual_readout="не (mean pool)", batchnorm="да",
         params=344327, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="да (in-fold)", dropedge="не",
         acc=0.5453, acc_std=0.0660, f1=0.4579, f1_std=0.0797,
         checkpoint="expA_gatconv_v4.pt", verified=True,
         metrics_source="expA_results.txt / baseline_results.txt", verified_ckpt=True,
         tag="GATConv repro",
         whats_new="Репродукция на GATConv с друг seed (head-to-head проверка на стабилност). Най-висок F1 сред 160-генните GATConv варианти."),

    dict(version_id="V6", label="V6 · 160 гена\nGATv2 (fixed edge)", date="2026-06-08",
         script="train (Colab)", data_source="TCGA", n_patients=981,
         n_genes=160, layer_type="GATv2Conv", n_layers=3, hidden=128, heads=4,
         node_feat=1, edge_used="да [0.4–1.0] (поправено)", jk="не", dual_readout="не (mean pool)", batchnorm="да",
         params=674823, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="да (in-fold)", dropedge="не",
         acc=0.6136, acc_std=0.0302, f1=0.4544, f1_std=0.0561,
         checkpoint="best_model_tcga_160genes_06,08,26.pt", verified=True,
         metrics_source="Материали §10 / baseline_results_196genes.txt", verified_ckpt=True,
         tag="edge поправен",
         whats_new="Поправка на двойната нормализация → реален STRING combined_score в attention (edge_dim=1). +6.0 pp спрямо V5; потвърждава, че GATv2 помага САМО с реален ребрен сигнал."),

    dict(version_id="196g", label="196 гена\nGATv2 3L  (★ user-best)", date="2026-06-09",
         script="train_tcga_only.py (196г)", data_source="TCGA", n_patients=981,
         n_genes=196, layer_type="GATv2Conv", n_layers=3, hidden=128, heads=4,
         node_feat=1, edge_used="да [0.4–1.0]", jk="не", dual_readout="не (mean pool)", batchnorm="да",
         params=674823, loss="CE (weighted)", optimizer="Adam",
         scheduler="ReduceLROnPlateau", smote="да (in-fold)", dropedge="не",
         acc=0.6861, acc_std=0.0201, f1=0.5429, f1_std=0.0611,
         checkpoint="best_model_tcga_196genes_09.06.pt", verified=True,
         metrics_source="training_log(1).txt / baseline_results_196genes.txt", verified_ckpt=True,
         tag="+36 PAM50 → 196г",
         whats_new="Добавени 36 PAM50 гена (160→196) за по-добро LumA/LumB разграничаване. +7.3 pp Acc. Посочен от автора като production-модел за деплой/UI."),

    dict(version_id="BioGNNv4", label="BioGNNv4 Enhanced\n196 гена · JK+dual", date="2026-06-17",
         script="train_colab_v4.py", data_source="TCGA", n_patients=981,
         n_genes=196, layer_type="GATv2Conv", n_layers=3, hidden=128, heads=4,
         node_feat=3, edge_used="да [0.4–1.0]", jk="да", dual_readout="да (mean‖max)", batchnorm="да (bn1,bn2,bn3)",
         params=964488, loss="FocalLoss(γ=2.0)", optimizer="AdamW",
         scheduler="CosineAnnealingWarmRestarts", smote="да (in-fold)", dropedge="не",
         acc=0.6932, acc_std=0.0250, f1=0.6646, f1_std=0.0240,
         checkpoint="last_model-17-06.pt", verified=True,
         metrics_source="v4_results.txt", verified_ckpt=True,
         tag="JK+dual+Focal+3feat",
         whats_new="3 node features [z, degree, mean_edge] + JumpingKnowledge + dual readout (mean‖max) + FocalLoss(γ=2) + AdamW + CosineAnnealing. Най-висок GNN Acc И F1 (драстичен скок на F1: 0.54→0.66)."),

    dict(version_id="v5DE", label="BioGNNv4 + DropEdge\n196 гена", date="2026-06-18",
         script="train_colab_v5.py", data_source="TCGA", n_patients=981,
         n_genes=196, layer_type="GATv2Conv", n_layers=3, hidden=128, heads=4,
         node_feat=3, edge_used="да [0.4–1.0]", jk="да", dual_readout="да (mean‖max)", batchnorm="да",
         params=964488, loss="FocalLoss(γ=1.0)", optimizer="AdamW",
         scheduler="CosineAnnealingWarmRestarts", smote="да (in-fold)", dropedge="да (p=0.15)",
         acc=0.69, acc_std=None, f1=0.66, f1_std=None,
         checkpoint="(няма запазен .pt)", verified=False,
         metrics_source="CLAUDE.md (приблизителни)", verified_ckpt=False,
         tag="+DropEdge",
         whats_new="BioGNNv4 + DropEdge(p=0.15) + FocalLoss(γ=1.0) за регуляризация. Резултатите ~равни на BioGNNv4; чекпойнтът не е намерен на диска (не може да се верифицира)."),
]

# Baselines (за референтни линии)
BASELINES = {
    "Majority class":      dict(acc=0.5087, f1=0.1349, color="#9aa0a6"),
    "Logistic Regression": dict(acc=0.8960, f1=0.8552, color="#1a73e8"),
    "Random Forest":       dict(acc=0.9072, f1=0.7896, color="#188038"),
    "MLP":                 dict(acc=0.8919, f1=0.8392, color="#e37400"),
}

# Per-class F1 (само версии с налични данни). inflated=True → завишени (full-dataset)
PERCLASS = {
    "Exp A (GATConv 160г)":   dict(vals=[0.7266, 0.3394, 0.6559, 0.3500, 0.2588], inflated=False),
    "V6 (GATv2 160г)*":       dict(vals=[0.87, 0.62, 0.78, 0.44, 0.69],            inflated=True),
    "196г GATv2 (★)":         dict(vals=[0.85, 0.40, 0.81, 0.59, 0.60],            inflated=False),
    "BioGNNv4 Enhanced":      dict(vals=[0.9547, 0.6750, 0.6922, 0.6176, 0.3817],  inflated=False),
}
CLASS_NAMES = ["Basal", "Her2", "LumA", "LumB", "Normal"]

# Confusion matrix — BioGNNv4 Enhanced (от v4_results.txt, combined CV folds)
CM_BIOGNNV4 = np.array([
    [158,  3,   1,   7,   2],
    [  0, 54,   0,  24,   0],
    [  0,  6, 271, 156,  66],
    [  2, 17,   4, 172,   2],
    [  0,  2,   8,   1,  25],
])

# Ablation waterfall: (label, acc_from, acc_to, added_component)
WATERFALL = [
    ("V1 (27г)",        None,   0.5658, "Базов прототип GATConv"),
    ("→ V4 (160г)",     0.5658, 0.5851, "+160 гена, per-patient норм., SMOTE, early stop"),
    ("→ V5 (GATv2)",    0.5851, 0.5535, "+GATv2 +edge (БЪГ: edge≈0)"),
    ("→ V6 (fixed)",    0.5535, 0.6136, "поправка edge норм. (реален STRING сигнал)"),
    ("→ 196г",          0.6136, 0.6861, "+36 PAM50 гена (160→196)"),
    ("→ BioGNNv4",      0.6861, 0.6932, "+3 node feat, JK, dual readout, FocalLoss, AdamW"),
]

# ============================================================================
# 2. CSV + конзолен преглед
# ============================================================================
csv_cols = ["version_id", "date", "script", "data_source", "n_patients", "n_genes",
            "layer_type", "n_layers", "hidden", "heads", "node_feat", "edge_used",
            "jk", "dual_readout", "batchnorm", "params", "loss", "optimizer",
            "scheduler", "smote", "dropedge", "acc", "acc_std", "f1", "f1_std",
            "checkpoint", "verified_ckpt", "metrics_source", "whats_new"]
rows = []
for v in VERSIONS:
    rows.append({c: v.get(c) for c in csv_cols})
df = pd.DataFrame(rows, columns=csv_cols)
csv_path = os.path.join(ROOT, "model_versions_summary.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("=" * 100)
print("ОБОБЩИТЕЛНА ТАБЛИЦА НА ВЕРСИИТЕ (за ревизия)")
print("=" * 100)
show = df[["version_id", "date", "n_genes", "layer_type", "node_feat", "edge_used",
           "jk", "params", "acc", "f1", "verified_ckpt", "checkpoint"]]
with pd.option_context("display.max_rows", None, "display.width", 200,
                       "display.max_colwidth", 34):
    print(show.to_string(index=False))
print("\nCSV записан:", csv_path)

# ============================================================================
# 3. ФИГУРИ + PDF
# ============================================================================
SUBT = "BRCA Subtype GNN · сравнение на версиите"
ORDER = [v for v in VERSIONS]           # вече хронологичен
xlabels = [v["label"] for v in ORDER]
accs = [v["acc"] for v in ORDER]
f1s = [v["f1"] for v in ORDER]
acc_err = [v["acc_std"] if v["acc_std"] else 0 for v in ORDER]
f1_err = [v["f1_std"] if v["f1_std"] else 0 for v in ORDER]
bar_colors = ["#c5221f" if not v["verified_ckpt"] else "#1a73e8" for v in ORDER]

def caption(fig, text):
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=8.5,
             style="italic", wrap=True, color="#3c4043")

pdf_path = os.path.join(ROOT, "model_versions_report.pdf")
pdf = PdfPages(pdf_path)

# ---- Заглавна страница ------------------------------------------------------
fig = plt.figure(figsize=(11.69, 8.27))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.5, 0.80, "Сравнителен анализ на версиите на модела",
        ha="center", fontsize=24, fontweight="bold", color="#202124")
ax.text(0.5, 0.72, "BRCA Subtype GNN — класификация на 5 молекулярни подтипа на рак на гърдата",
        ha="center", fontsize=13, color="#3c4043")
n_ver = len(VERSIONS)
n_ck = sum(1 for v in VERSIONS if v["verified_ckpt"])
n_files = 7
ax.text(0.5, 0.58,
        f"Анализирани версии: {n_ver}\n"
        f"Верифицирани чекпойнти (презаредени .pt): {n_ck} от {n_files} .pt файла\n"
        f"Данни: TCGA brca_tcga_pan_can_atlas_2018 · 981 пациента · 27→160→196 гена",
        ha="center", fontsize=12, color="#202124", linespacing=1.7)
ax.text(0.5, 0.34,
        "РЕЗЮМЕ: Най-добрият GNN по метрики е BioGNNv4 Enhanced (Acc=0.693, Macro F1=0.665,\n"
        "верифициран чекпойнт last_model-17-06.pt). Авторът посочва за production-модел\n"
        "196-генния GATv2 (best_model_tcga_196genes_09.06.pt, Acc=0.686, F1=0.543) — по-консервативен\n"
        "и интегриран в UI. ВАЖНА УГОВОРКА: табличните baseline-и (Random Forest Acc=0.907,\n"
        "Logistic Regression Acc=0.896 / F1=0.855) все още значимо превъзхождат всеки GNN —\n"
        "това е честният методологичен извод на проекта.",
        ha="center", fontsize=10.5, color="#3c4043", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.8", fc="#fef7e0", ec="#f9ab00"))
ax.text(0.5, 0.08, f"Генериран автоматично · {GEN_DATE}", ha="center",
        fontsize=10, color="#5f6368")
pdf.savefig(fig); plt.close(fig)

# ---- Обобщителна таблица (като фигура) -------------------------------------
fig = plt.figure(figsize=(16.5, 8.27)); ax = fig.add_subplot(111); ax.axis("off")
ax.set_title("Обобщителна таблица на всички версии", fontsize=15, fontweight="bold", pad=12)
tbl_cols = ["Версия", "Дата", "Гени", "Слой", "node\nfeat", "edge", "JK", "dual\nRO",
            "Параметри", "Loss", "Optim", "SMOTE", "Acc", "Macro F1", "Verified\nckpt"]
tdata = []
for v in VERSIONS:
    tdata.append([
        v["version_id"], v["date"], v["n_genes"], v["layer_type"], v["node_feat"],
        "да" if "да" in str(v["edge_used"]) else "не", v["jk"],
        "да" if "да" in str(v["dual_readout"]) else "не",
        f"{v['params']:,}" if v["params"] else "—",
        v["loss"].split("(")[0], v["optimizer"], v["smote"].split(" ")[0],
        f"{v['acc']:.4f}" + ("~" if v["acc_std"] is None and not v["verified_ckpt"] else ""),
        f"{v['f1']:.4f}",
        "✔" if v["verified_ckpt"] else "✘",
    ])
the_table = ax.table(cellText=tdata, colLabels=tbl_cols, cellLoc="center", loc="center")
the_table.auto_set_font_size(False); the_table.set_fontsize(8.2); the_table.scale(1, 1.7)
for (r, c), cell in the_table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1a73e8"); cell.set_text_props(color="white", fontweight="bold")
    else:
        v = VERSIONS[r - 1]
        cell.set_facecolor("#e6f4ea" if v["verified_ckpt"] else "#fce8e6")
caption(fig, "Сини/зелени редове = архитектурата е верифицирана чрез презареждане на .pt чекпойнта. "
             "Червени = само текстов източник (липсва/невъзможен за презареждане чекпойнт). "
             "'~' = приблизителна стойност.")
pdf.savefig(fig); plt.close(fig)

# ---- A. Времева линия -------------------------------------------------------
fig, ax = plt.subplots(figsize=(16.5, 7.0))
xs = list(range(len(ORDER)))
ax.axhline(0, color="#9aa0a6", lw=2, zorder=1)
for i, v in enumerate(ORDER):
    col = "#1a73e8" if v["verified_ckpt"] else "#c5221f"
    ax.scatter(i, 0, s=180, color=col, zorder=3, edgecolor="white", linewidth=1.5)
    up = (i % 2 == 0)
    y = 0.45 if up else -0.45
    va = "bottom" if up else "top"
    ax.annotate(f"{v['version_id']}\n{v['date']}\nAcc={v['acc']:.3f}  F1={v['f1']:.3f}",
                xy=(i, 0), xytext=(i, y), ha="center", va=va, fontsize=8.5,
                fontweight="bold", color=col,
                arrowprops=dict(arrowstyle="-", color=col, lw=1))
    y2 = 1.02 if up else -1.02
    ax.text(i, y2, v.get("tag", ""), ha="center", va=va, fontsize=8,
            color="#3c4043", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f1f3f4", ec="#dadce0", lw=0.5))
ax.set_xlim(-0.6, len(ORDER) - 0.4); ax.set_ylim(-1.6, 1.6)
ax.axis("off")
ax.set_title("A. Времева линия на версиите — какво ново добавя всяка",
             fontsize=14, fontweight="bold")
caption(fig, "Синьо = верифициран чекпойнт · Червено = само документиран (без чекпойнт). "
             "Хронологичен ред отляво надясно.")
pdf.savefig(fig); plt.close(fig)

# ---- B. CV Accuracy bar -----------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.bar(xs, accs, yerr=acc_err, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
for i, a in enumerate(accs):
    ax.text(i, a + (acc_err[i] or 0) + 0.012, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
bl = [("Random Forest", "#188038"), ("Logistic Regression", "#1a73e8"), ("Majority class", "#9aa0a6")]
for name, col in bl:
    ax.axhline(BASELINES[name]["acc"], color=col, ls="--", lw=1.6,
               label=f"{name} (baseline) = {BASELINES[name]['acc']:.3f}")
ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("CV Accuracy"); ax.set_ylim(0, 1.0)
ax.set_title("B. CV Accuracy по версии (с error bars, хронологично)", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)
caption(fig, "Сини стълбове = верифициран чекпойнт; червени = само документиран. "
             "Прекъснатите линии са табличните baseline-и — всички GNN остават под тях.")
fig.tight_layout(rect=[0, 0.05, 1, 1]); pdf.savefig(fig); plt.close(fig)

# ---- C. CV Macro F1 bar -----------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.bar(xs, f1s, yerr=f1_err, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
for i, a in enumerate(f1s):
    ax.text(i, a + (f1_err[i] or 0) + 0.012, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
for name, col in [("Logistic Regression", "#1a73e8"), ("Random Forest", "#188038"),
                  ("MLP", "#e37400"), ("Majority class", "#9aa0a6")]:
    ax.axhline(BASELINES[name]["f1"], color=col, ls="--", lw=1.6,
               label=f"{name} = {BASELINES[name]['f1']:.3f}")
ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("CV Macro F1"); ax.set_ylim(0, 1.0)
ax.set_title("C. CV Macro F1 по версии (с error bars)", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)
caption(fig, "Macro F1 третира всичките 5 класа равностойно — по-честна метрика при дисбаланса 13.9×. "
             "Скокът при BioGNNv4 (0.54→0.66) идва от FocalLoss + dual readout.")
fig.tight_layout(rect=[0, 0.05, 1, 1]); pdf.savefig(fig); plt.close(fig)

# ---- D. Accuracy vs Macro F1 рамо до рамо ----------------------------------
fig, ax = plt.subplots(figsize=(14, 7.5))
w = 0.38
ax.bar([x - w/2 for x in xs], accs, w, label="Accuracy", color="#1a73e8", edgecolor="black", linewidth=0.5)
ax.bar([x + w/2 for x in xs], f1s, w, label="Macro F1", color="#e37400", edgecolor="black", linewidth=0.5)
for i in xs:
    ax.text(i - w/2, accs[i] + 0.008, f"{accs[i]:.2f}", ha="center", fontsize=7.5)
    ax.text(i + w/2, f1s[i] + 0.008, f"{f1s[i]:.2f}", ha="center", fontsize=7.5)
    gap = accs[i] - f1s[i]
    ax.text(i, max(accs[i], f1s[i]) + 0.05, f"Δ={gap:+.2f}", ha="center", fontsize=7.5,
            color="#c5221f" if gap > 0.12 else "#188038", fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("Стойност"); ax.set_ylim(0, 0.95)
ax.set_title("D. Accuracy vs Macro F1 рамо до рамо (Δ = Acc − F1)", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10); ax.grid(axis="y", alpha=0.3)
caption(fig, "Голямо Δ (червено) = моделът разчита на честия LumA, но се проваля при редките класове. "
             "BioGNNv4 има най-малко Δ (0.03) → най-балансиран модел.")
fig.tight_layout(rect=[0, 0.05, 1, 1]); pdf.savefig(fig); plt.close(fig)

# ---- E. Per-class F1 heatmap ------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))
mat = np.array([PERCLASS[k]["vals"] for k in PERCLASS])
rows_lbl = list(PERCLASS.keys())
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if mat[i, j] < 0.45 else "black")
ax.set_title("E. Per-class F1 по версии (червено=слабо, зелено=силно)", fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="F1")
caption(fig, "* V6 редът е завишен (оценен на full dataset, вкл. train). "
             "Виж как Normal и LumB остават проблемни; BioGNNv4 рязко вдига Her2 и LumB, но сваля Normal.")
fig.tight_layout(rect=[0, 0.05, 1, 1]); pdf.savefig(fig); plt.close(fig)

# ---- F. Ablation waterfall --------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7.5))
running = WATERFALL[0][2]
xpos = 0
ax.bar(xpos, running, color="#5f6368", edgecolor="black", zorder=3)
ax.text(xpos, running + 0.01, f"{running:.3f}", ha="center", fontsize=9, fontweight="bold")
labels_wf = [WATERFALL[0][0]]
for (lbl, a_from, a_to, comp) in WATERFALL[1:]:
    xpos += 1
    delta = a_to - a_from
    col = "#188038" if delta >= 0 else "#c5221f"
    ax.bar(xpos, delta, bottom=a_from, color=col, edgecolor="black", zorder=3)
    ax.plot([xpos - 1.4, xpos + 0.4], [a_from, a_from], color="#9aa0a6", ls=":", lw=1, zorder=2)
    ytxt = a_to + 0.012 if delta >= 0 else a_from + 0.012
    ax.text(xpos, ytxt, f"{delta:+.3f}", ha="center", fontsize=9, fontweight="bold", color=col)
    labels_wf.append(lbl)
ax.set_xticks(range(len(labels_wf))); ax.set_xticklabels(labels_wf, fontsize=9)
ax.set_ylabel("CV Accuracy"); ax.set_ylim(0.45, 0.75)
ax.set_title("F. Каскадна (waterfall) диаграма: какво добавя всеки преход → Δ Accuracy",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
comp_txt = "\n".join(f"{w[0]:<14s} {w[3]}" for w in WATERFALL)
ax.text(0.015, 0.97, comp_txt, transform=ax.transAxes, fontsize=7.6, va="top",
        family="DejaVu Sans", bbox=dict(boxstyle="round", fc="#f1f3f4", ec="#dadce0"))
caption(fig, "Зелено = подобрение, червено = регресия. Най-големите печалби: поправка на edge-бъга (+0.060) "
             "и добавяне на 36 PAM50 гена (+0.073). Архитектурните трикове в BioGNNv4 дават малко Acc, но много F1.")
fig.tight_layout(rect=[0, 0.05, 1, 1]); pdf.savefig(fig); plt.close(fig)

# ---- G. Confusion matrix (BioGNNv4) ----------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7.5))
cm = CM_BIOGNNV4
cmn = cm / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Предсказано"); ax.set_ylabel("Истина")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.0f}%", ha="center", va="center",
                fontsize=10, color="white" if cmn[i, j] > 0.5 else "black")
ax.set_title("G. Confusion matrix — BioGNNv4 Enhanced (combined CV, 981 пациента)",
             fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="дял по ред")
caption(fig, "Единственият модел с реална CM в result-файловете. Вижда се основното objuркване: "
             "LumA→LumB (156) и LumA→Normal (66). Basal се разпознава почти перфектно (158/171).")
fig.tight_layout(rect=[0, 0.05, 1, 1]); pdf.savefig(fig); plt.close(fig)

# ---- H. Discrepancy panel ---------------------------------------------------
fig = plt.figure(figsize=(16.5, 9.0)); ax = fig.add_subplot(111); ax.axis("off")
ax.set_title("H. Discrepancy panel — разминавания текст ↔ чекпойнт",
             fontsize=15, fontweight="bold", pad=10)
disc = [
    ["#", "Разминаване", "Текстов източник твърди", "Верификация от чекпойнт / реалност"],
    ["1", "Кой е 'най-добрият' модел",
     "v4_results.txt: BioGNNv4 Acc=0.6932 (най-добър).\nCLAUDE.md/автор: 196g_09.06 е production.",
     "Различни модели! BioGNNv4 (JK,3feat) има по-висок\nAcc И F1. 196g_09.06 (3L GATv2,1feat) е по-консерв."],
    ["2", "BioGNNv4 чекпойнт име",
     "v4_results.txt: './best_model_v4_196genes.pt'\nCLAUDE.md: best_model_v4_196genes.pt",
     "Такъв файл ЛИПСВА. Реалният чекпойнт е\nlast_model-17-06.pt (964,488 пар., JK+dual — съвпада)."],
    ["3", "best_model_v5_196genes.pt",
     "CLAUDE.md го изброява (DropEdge, ~0.69).",
     "ЛИПСВА на диска. v5DE не може да се верифицира."],
    ["4", "V5 vs V6 архитектура",
     "Материали §10: V6 е 'нова архитектура' спрямо V5.",
     "Идентични state_dict форми (674,823 пар.). Разликата е\nСАМО edge-норм. бъгфикс в data_loader — невидим в теглата."],
    ["5", "Брой параметри GATConv",
     "Материали §10.1: GATConv V4 ~480K параметри.",
     "Верифицирано: 344,327 параметри (GATv2=674,823 ✓)."],
    ["6", "Брой гени (етикети)",
     "CLAUDE.md: 196 · baseline_results_196genes.txt: 184\nbaseline_results.txt: 157",
     "PPI графът има 4180 ребра при 196-генните рунове;\nразлики идват от 'покрити в STRING' vs 'в панела'."],
    ["7", "196g чекпойнт = BioGNNv4?",
     "Подсказва се, че 196г е 'enhanced'.",
     "НЕ. 196g_09.06 = 3L GATv2, 1 node feat (conv1.lin_l=512×1),\nlin1=128×64. Не е JK/dual. Различен от BioGNNv4."],
]
tbl = ax.table(cellText=disc[1:], colLabels=disc[0], cellLoc="left", loc="center",
               colWidths=[0.03, 0.18, 0.36, 0.40])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.0); tbl.scale(1, 3.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_text_props(va="center")
    if r == 0:
        cell.set_facecolor("#c5221f"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 3:
        cell.set_facecolor("#e6f4ea")
    elif c == 2:
        cell.set_facecolor("#fef7e0")
ax.text(0.5, 0.045,
        "★ Реален best според автора: best_model_tcga_196genes_09.06.pt (Acc=0.686, F1=0.543) — "
        "production-моделът в UI. Най-добър по метрики: BioGNNv4 / last_model-17-06.pt (Acc=0.693, F1=0.665).",
        transform=ax.transAxes, ha="center", fontsize=9.5, fontweight="bold", color="#202124",
        bbox=dict(boxstyle="round,pad=0.6", fc="#e8f0fe", ec="#1a73e8"))
pdf.savefig(fig); plt.close(fig)

# ---- Приложение: статус на чекпойнтите -------------------------------------
fig = plt.figure(figsize=(16.5, 8.27)); ax = fig.add_subplot(111); ax.axis("off")
ax.set_title("Приложение · Списък на .pt чекпойнтите и статус на верификацията",
             fontsize=14, fontweight="bold", pad=12)
ck = [
    ["Чекпойнт файл", "Размер", "Параметри", "Архитектура (верифицирана)", "Версия", "Статус"],
    ["best_model_tcga_160genes.pt", "1.39 MB", "344,327", "GATConv 3L, 1 feat, mean pool", "V4", "✔ verified"],
    ["Last_model_tcga_160genes(1).pt", "2.71 MB", "674,823", "GATv2 3L +edge, 1 feat", "V5 (buggy)", "✔ verified"],
    ["expA_gatconv_v4.pt", "1.39 MB", "344,327", "GATConv 3L, 1 feat", "Exp A", "✔ verified"],
    ["expA_gatconv_v4(1).pt", "1.39 MB", "344,327", "GATConv 3L (дубликат)", "Exp A", "✔ verified"],
    ["best_model_tcga_160genes_06,08,26.pt", "2.71 MB", "674,823", "GATv2 3L +edge, 1 feat", "V6 (fixed)", "✔ verified"],
    ["best_model_tcga_196genes_09.06.pt", "2.71 MB", "674,823", "GATv2 3L +edge, 1 feat", "196g ★", "✔ verified"],
    ["last_model-17-06.pt", "3.87 MB", "964,488", "GATv2 3L +3feat +JK +dual +bn3", "BioGNNv4", "✔ verified"],
    ["best_model_v4_196genes.pt", "—", "—", "посочен в текста, но липсва", "(BioGNNv4?)", "✘ липсва"],
    ["best_model_v5_196genes.pt", "—", "—", "посочен в CLAUDE.md, но липсва", "(v5 DropEdge?)", "✘ липсва"],
]
tbl = ax.table(cellText=ck[1:], colLabels=ck[0], cellLoc="left", loc="center",
               colWidths=[0.30, 0.08, 0.10, 0.30, 0.12, 0.10])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.6); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1a73e8"); cell.set_text_props(color="white", fontweight="bold")
    else:
        ok = "verified" in ck[r][5]
        cell.set_facecolor("#e6f4ea" if ok else "#fce8e6")
ax.text(0.5, 0.07,
        "7 .pt файла на диска — всичките 7 успешно презаредени и инспектирани (state_dict форми → архитектура). "
        "2 файла, цитирани в документацията, не съществуват.",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#3c4043")
pdf.savefig(fig); plt.close(fig)

pdf.close()
print("\nPDF записан:", pdf_path)
print("Страници: заглавна + таблица + A–H + приложение =", 11)
