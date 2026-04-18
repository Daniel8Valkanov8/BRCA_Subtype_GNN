"""
Обучение с 160 гена, TCGA + METABRIC (2737 пациента), подобрена архитектура.
TensorBoard: стартирай с -> venv/Scripts/tensorboard --logdir=runs
"""
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

from src.data_loader import build_graph_dataset
from src.model import BioGNN

# --- КОНФИГУРАЦИЯ ---
HIDDEN_CHANNELS = 128
EPOCHS          = 150
BATCH_SIZE      = 16
LR              = 0.001
WEIGHT_DECAY    = 1e-4

# --- ЗАРЕЖДАНЕ НА ДАННИ ---
print("Зареждане на данни...")

# TCGA
clinical_df = pd.read_csv('data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
                          sep='\t', comment='#').dropna(subset=['Subtype'])
tcga_expr   = pd.read_csv('data/tcga_expression_174genes.csv', index_col=0)
common_tcga = sorted(set(clinical_df['Sample ID']) & set(tcga_expr.columns))
tcga_expr   = tcga_expr[common_tcga].fillna(0)
tcga_clin   = clinical_df.set_index('Sample ID').loc[common_tcga, ['Subtype']]

# METABRIC
meta_expr = pd.read_csv('data/metabric_expression_160genes.csv', index_col=0)
meta_clin = pd.read_csv('data/metabric_clinical.csv', index_col=0)
common_meta = sorted(set(meta_expr.columns) & set(meta_clin.index))
meta_expr = meta_expr[common_meta].fillna(0)
meta_clin = meta_clin.loc[common_meta, ['Subtype']]

# Общи гени между двете бази
common_genes = sorted(set(tcga_expr.index) & set(meta_expr.index))
tcga_expr    = tcga_expr.loc[common_genes]
meta_expr    = meta_expr.loc[common_genes]

# Комбиниране
exp_combined  = pd.concat([tcga_expr, meta_expr], axis=1)
clin_combined = pd.concat([tcga_clin, meta_clin])

print(f"TCGA:     {tcga_expr.shape[1]} пациента")
print(f"METABRIC: {meta_expr.shape[1]} пациента")
print(f"Общо:     {exp_combined.shape[1]} пациента x {len(common_genes)} гена")
print(f"\nРазпределение:")
print(clin_combined['Subtype'].value_counts())

# PPI мрежа
ppi_df = pd.read_csv('data/string_ppi_174genes.tsv', sep='\t')

# Изграждане на графове
print("\nИзграждане на графове...")
dataset, encoder = build_graph_dataset(exp_combined, ppi_df, clin_combined)
print(f"Графове: {len(dataset)}")

# --- ПРЕТЕГЛЕН LOSS ---
device  = torch.device('cpu')
counts  = clin_combined['Subtype'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = weights / weights.sum() * len(counts)
print(f"\nClass weights: { {c: round(w,3) for c,w in zip(encoder.classes_, weights.tolist())} }")

# --- 5-FOLD CROSS VALIDATION ---
all_labels = [g.y.item() for g in dataset]
skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_f1s = [], []
best_acc, best_state = 0.0, None

print(f"\nСтартиране на 5-Fold CV (hidden={HIDDEN_CHANNELS}, epochs={EPOCHS})...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), all_labels), 1):
    print(f"{'='*50}")
    print(f"FOLD {fold}/5")
    print(f"{'='*50}")

    writer = SummaryWriter(log_dir=f'runs/fold_{fold}')

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader([dataset[i] for i in test_idx],  batch_size=BATCH_SIZE)

    model     = BioGNN(1, HIDDEN_CHANNELS, len(encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    for epoch in range(1, EPOCHS + 1):
        # Обучение
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Валидация след всяка епоха
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                preds.extend(out.argmax(1).numpy())
                trues.extend(batch.y.numpy())

        val_acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        val_f1  = f1_score(trues, preds, average='macro', zero_division=0)

        scheduler.step(avg_loss)

        # TensorBoard логване
        writer.add_scalar('Loss/train',    avg_loss, epoch)
        writer.add_scalar('Accuracy/val',  val_acc,  epoch)
        writer.add_scalar('F1_macro/val',  val_f1,   epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Принтираме на всеки 10 епохи
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    writer.close()

    fold_accs.append(val_acc)
    fold_f1s.append(val_f1)
    print(f"\n  Fold {fold} финал -> Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}")

    if val_acc > best_acc:
        best_acc   = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  --> Нов най-добър модел (acc={best_acc:.4f})")

torch.save(best_state, 'best_model_v2.pt')

print(f"\n{'='*50}")
print(f"ФИНАЛНИ РЕЗУЛТАТИ (160 гена, 2737 пациента)")
print(f"{'='*50}")
print(f"Accuracy:  {sum(fold_accs)/5:.4f}  (+/- {pd.Series(fold_accs).std():.4f})")
print(f"Macro F1:  {sum(fold_f1s)/5:.4f}  (+/- {pd.Series(fold_f1s).std():.4f})")
print(f"\nСРАВНЕНИЕ:")
print(f"  27 гена оригинал:  Acc=0.5658  F1=0.4591")
print(f"  160 гена v2:       Acc={sum(fold_accs)/5:.4f}  F1={sum(fold_f1s)/5:.4f}")

# Per-class report
model.load_state_dict(best_state)
model.eval()
p_all, t_all = [], []
with torch.no_grad():
    for batch in DataLoader(dataset[:len(common_tcga)], batch_size=32):
        out = model(batch.x, batch.edge_index, batch.batch)
        p_all.extend(out.argmax(1).numpy())
        t_all.extend(batch.y.numpy())

print("\nPer-class report (тест само върху TCGA пациенти):")
print(classification_report(t_all, p_all, target_names=encoder.classes_, zero_division=0))
print("\nМоделът е запазен като: best_model_v2.pt")
print("TensorBoard логовете са в папка: runs/")
