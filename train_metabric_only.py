"""
Обучение само на METABRIC данни · 160 гена · 5-Fold CV
"""
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

from src.data_loader import build_graph_dataset
from src.model import BioGNN

HIDDEN_CHANNELS = 128
EPOCHS          = 150
BATCH_SIZE      = 16
LR              = 0.0005
WEIGHT_DECAY    = 1e-4
PATIENCE        = 20
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print("Зареждане на METABRIC данни...")

meta_expr = pd.read_csv('data/metabric_expression_160genes.csv', index_col=0)
meta_clin = pd.read_csv('data/metabric_clinical.csv', index_col=0)
common    = sorted(set(meta_expr.columns) & set(meta_clin.index))
meta_expr = meta_expr[common].fillna(0)
meta_clin = meta_clin.loc[common, ['Subtype']]

# Филтрираме само BRCA подтипове (същите като TCGA)
valid_subtypes = ['BRCA_Basal', 'BRCA_Her2', 'BRCA_LumA', 'BRCA_LumB', 'BRCA_Normal']
mask      = meta_clin['Subtype'].isin(valid_subtypes)
meta_clin = meta_clin[mask]
meta_expr = meta_expr[meta_clin.index]

# Per-patient z-score нормализация
print("Нормализация (per-patient z-score)...")
meta_expr = meta_expr.apply(lambda col: (col - col.mean()) / (col.std() + 1e-8), axis=0)

print(f"Пациенти: {meta_expr.shape[1]}  |  Гени: {meta_expr.shape[0]}")
print(f"\nРазпределение:\n{meta_clin['Subtype'].value_counts()}\n")

ppi_df   = pd.read_csv('data/string_interactions_short.tsv', sep='\t', comment='#')
dataset, encoder = build_graph_dataset(meta_expr, ppi_df, meta_clin)
print(f"Графове: {len(dataset)}  |  Ребра: {dataset[0].edge_index.shape[1]}")

counts  = meta_clin['Subtype'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = (weights / weights.sum() * len(counts)).to(DEVICE)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_labels = [g.y.item() for g in dataset]
fold_accs, fold_f1s = [], []
best_acc, best_state = 0.0, None

print(f"Стартиране на 5-Fold CV (hidden={HIDDEN_CHANNELS}, epochs={EPOCHS}, lr={LR})...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), all_labels), 1):
    print(f"{'='*50}\nFOLD {fold}/5\n{'='*50}")
    writer = SummaryWriter(log_dir=f'runs/metabric_fold_{fold}')

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader([dataset[i] for i in test_idx],  batch_size=BATCH_SIZE)

    model     = BioGNN(1, HIDDEN_CHANNELS, len(encoder.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_fold_acc   = 0.0
    best_fold_state = None
    no_improve      = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                out   = model(batch.x, batch.edge_index, batch.batch)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(batch.y.cpu().numpy())

        val_acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        val_f1  = f1_score(trues, preds, average='macro', zero_division=0)
        scheduler.step(avg_loss)

        writer.add_scalar('Loss/train',   avg_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc,  epoch)
        writer.add_scalar('F1_macro/val', val_f1,   epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if val_acc > best_fold_acc:
            best_fold_acc   = val_acc
            best_fold_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve      = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Best: {best_fold_acc:.4f}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping на epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    writer.close()
    fold_accs.append(best_fold_acc)

    model.load_state_dict(best_fold_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out   = model(batch.x, batch.edge_index, batch.batch)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(batch.y.cpu().numpy())
    best_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    fold_f1s.append(best_f1)

    print(f"\n  Fold {fold} Best -> Accuracy: {best_fold_acc:.4f} | Macro F1: {best_f1:.4f}")

    if best_fold_acc > best_acc:
        best_acc   = best_fold_acc
        best_state = best_fold_state
        print(f"  --> Нов глобален най-добър модел (acc={best_acc:.4f})")

torch.save(best_state, 'best_model_metabric_160genes.pt')

print(f"\n{'='*50}")
print(f"ФИНАЛНИ РЕЗУЛТАТИ — METABRIC ONLY (160 гена, {meta_expr.shape[1]} пациента)")
print(f"{'='*50}")
print(f"Accuracy:  {sum(fold_accs)/5:.4f}  (+/- {pd.Series(fold_accs).std():.4f})")
print(f"Macro F1:  {sum(fold_f1s)/5:.4f}  (+/- {pd.Series(fold_f1s).std():.4f})")
print(f"\nСРАВНЕНИЕ:")
print(f"  27 гена оригинал:       Acc=0.5658  F1=0.4591")
print(f"  160 гена TCGA+META:     Acc=0.3745  F1=0.3562")
print(f"  160 гена METABRIC only: Acc={sum(fold_accs)/5:.4f}  F1={sum(fold_f1s)/5:.4f}")

model.load_state_dict(best_state)
model.eval()
p_all, t_all = [], []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=32):
        batch = batch.to(DEVICE)
        out   = model(batch.x, batch.edge_index, batch.batch)
        p_all.extend(out.argmax(1).cpu().numpy())
        t_all.extend(batch.y.cpu().numpy())

print("\nPer-class report:")
print(classification_report(t_all, p_all, target_names=encoder.classes_, zero_division=0))
print("Моделът е запазен като: best_model_metabric_160genes.pt")
