"""
Baseline сравнение: Majority Class, Logistic Regression, Random Forest, MLP
Използва идентична 5-Fold StratifiedKFold (random_state=42) и per-patient
z-score нормализация като GNN експериментите.

Локално: python baseline_comparison.py
Colab:   BASE = '/content/BRCA_Subtype_GNN' (вижте коментара по-долу)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# --- Пътища (промени BASE за Colab: BASE = '/content/BRCA_Subtype_GNN') ---
BASE      = '.'
# 196-генен набор (160 + 36 PAM50) — съвпада с последния/най-добър GNN модел (Acc=0.6861)
EXPR_PATH = os.path.join(BASE, 'data', 'tcga_expression_198genes.csv')
CLIN_PATH = os.path.join(BASE, 'data', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
PPI_PATH  = os.path.join(BASE, 'data', 'string_ppi_196genes.tsv')
OUT_PATH  = os.path.join(BASE, 'baseline_results_196genes.txt')

# --- Зареждане на данните ---
print("Зареждане на данни...")
for p in [EXPR_PATH, CLIN_PATH, PPI_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Липсва файл: {p}")

exp_df  = pd.read_csv(EXPR_PATH, index_col=0)
clin_df = pd.read_csv(CLIN_PATH, sep='\t', comment='#').dropna(subset=['Subtype'])
ppi_df  = pd.read_csv(PPI_PATH,  sep='\t')

# Напасване на пациенти (идентично на GNN pipeline)
common  = list(set(exp_df.columns) & set(clin_df['Sample ID']))
exp_df  = exp_df[common]
clin_df = clin_df.set_index('Sample ID').loc[common]
print(f"Пациенти: {len(common)}  |  Гени в CSV: {len(exp_df)}")

# Филтриране до PPI-покрити гени (идентично на GNN — само тези гени са nodes)
ppi_genes = set(ppi_df['node1']) | set(ppi_df['node2'])
gnn_genes = [g for g in exp_df.index if g in ppi_genes]
exp_gnn   = exp_df.loc[gnn_genes]
print(f"PPI-покрити гени (GNN nodes): {len(gnn_genes)}")

# Лейбъли
le      = LabelEncoder()
y       = le.fit_transform(clin_df['Subtype'])
classes = list(le.classes_)
print(f"Класове: {classes}")
dist    = dict(zip(classes, np.bincount(y)))
print(f"Разпределение: {dist}")

# Per-patient z-score нормализация (идентично на train_tcga_only.py)
X_raw = exp_gnn.values.T.astype(float)          # (981, n_genes)
mu    = X_raw.mean(axis=1, keepdims=True)
sd    = X_raw.std(axis=1,  keepdims=True) + 1e-8
X     = (X_raw - mu) / sd
print(f"Feature matrix: {X.shape}  (пациенти × гени)\n")

# --- Модели ---
models = {
    'Majority class': DummyClassifier(strategy='most_frequent'),

    'Logistic Regression': LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42,
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ),

    'MLP': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    ),
}

# --- 5-Fold CV ---
SKF     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"=== {name} ===")
    fold_acc, fold_f1       = [], []
    all_preds, all_true     = [], []

    for fold, (tr_idx, va_idx) in enumerate(SKF.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        acc = accuracy_score(y_va, preds)
        f1  = f1_score(y_va, preds, average='macro', zero_division=0)
        fold_acc.append(acc)
        fold_f1.append(f1)
        all_preds.extend(preds.tolist())
        all_true.extend(y_va.tolist())
        print(f"  Fold {fold+1}: acc={acc:.4f}  macro-f1={f1:.4f}")

    results[name] = {
        'acc_mean': np.mean(fold_acc),
        'acc_std':  np.std(fold_acc),
        'f1_mean':  np.mean(fold_f1),
        'f1_std':   np.std(fold_f1),
        'fold_acc': fold_acc,
        'fold_f1':  fold_f1,
        'report':   classification_report(
            all_true, all_preds,
            target_names=classes,
            digits=4,
        ),
    }
    print(f"  --> Acc: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}  "
          f"F1: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}\n")

# --- Обобщена таблица ---
HEADER = f"\n{'Модел':<25} {'Acc (mean ± std)':<22} {'Macro F1 (mean ± std)'}"
SEP    = '-' * 72

print(SEP)
print(HEADER)
print(SEP)
for name, r in results.items():
    print(f"{name:<25} {r['acc_mean']:.4f} ± {r['acc_std']:.4f}          "
          f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
print(SEP)
# GNN референтни стойности (5-Fold CV, идентичен random_state=42)
print(f"{'GNN 160г GATv2 (V4/v2)':<25} {'0.6136 ± ?':<22} {'0.4544 ± ?'}")
print(f"{'GNN 196г GATv2 (НАЙ-ДОБЪР)':<25} {'0.6861 ± 0.0201':<22} {'0.5429 ± 0.0611'}")
print(SEP)

# --- Запис ---
lines = []
lines.append("BASELINE COMPARISON — BRCA Subtype Classification")
lines.append("=" * 72)
lines.append(f"Данни: TCGA brca_tcga_pan_can_atlas_2018 | {len(common)} пациента | {len(gnn_genes)} гена")
lines.append(f"CV: 5-Fold StratifiedKFold (random_state=42)")
lines.append(f"Нормализация: per-patient z-score (идентично на GNN)")
lines.append("")
lines.append(HEADER)
lines.append(SEP)
for name, r in results.items():
    lines.append(
        f"{name:<25} {r['acc_mean']:.4f} ± {r['acc_std']:.4f}          "
        f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
    )
lines.append(SEP)
lines.append(f"{'GNN 160г GATv2 (V4/v2)':<25} {'0.6136 ± ?':<22} {'0.4544 ± ?'}")
lines.append(f"{'GNN 196г GATv2 (НАЙ-ДОБЪР)':<25} {'0.6861 ± 0.0201':<22} {'0.5429 ± 0.0611'}")
lines.append(SEP)
lines.append("")

for name, r in results.items():
    lines.append(f"\n=== {name} — Fold breakdown ===")
    for i, (a, f) in enumerate(zip(r['fold_acc'], r['fold_f1'])):
        lines.append(f"  Fold {i+1}: acc={a:.4f}  f1={f:.4f}")
    lines.append(f"\n=== {name} — Per-class F1 (combined validation) ===")
    lines.append(r['report'])

with open(OUT_PATH, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))

print(f"\nЗапазено: {OUT_PATH}")
