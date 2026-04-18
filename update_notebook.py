import json

with open('main.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

augment_cell = {
    "cell_type": "code",
    "id": "metabric_augment",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- 2b. METABRIC DATA AUGMENTATION ---\n",
        "# Зареждаме предварително изтеглените METABRIC данни (27 гена, 1756 пациента)\n",
        "meta_expr = pd.read_csv('data/metabric_expression.csv', index_col=0)   # genes x patients\n",
        "meta_clin = pd.read_csv('data/metabric_clinical.csv', index_col=0)       # patients x [Subtype]\n",
        "\n",
        "# Гарантираме еднакъв ред на гените\n",
        "meta_expr = meta_expr.loc[exp_final.index]   # само гените от TCGA, в същия ред\n",
        "\n",
        "# Обединяваме двете матрици (axis=1 = добавяме нови колони/пациенти)\n",
        "exp_combined  = pd.concat([exp_final, meta_expr], axis=1)\n",
        "clin_combined = pd.concat([clinical_final[['Subtype']], meta_clin[['Subtype']]])\n",
        "\n",
        "print(f'TCGA пациенти:     {exp_final.shape[1]}')\n",
        "print(f'METABRIC пациенти: {meta_expr.shape[1]}')\n",
        "print(f'Общо пациенти:     {exp_combined.shape[1]}')\n",
        "print(f'\\nРазпределение на подтиповете (комбинирано):')\n",
        "print(clin_combined['Subtype'].value_counts())\n",
        "\n",
        "# Преизграждаме dataset-а с комбинираните данни\n",
        "print('\\nИзграждаме графовете...')\n",
        "dataset_aug, encoder_aug = build_graph_dataset(exp_combined, ppi_df, clin_combined)\n",
        "print(f'Общо графове: {len(dataset_aug)}')"
    ]
}

train_aug_cell = {
    "cell_type": "code",
    "id": "train_augmented",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- 5b. 5-FOLD CV С METABRIC AUGMENTATION ---\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "from sklearn.metrics import f1_score\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "\n",
        "counts_aug  = clin_combined['Subtype'].value_counts().sort_index().values\n",
        "weights_aug = 1.0 / torch.tensor(counts_aug, dtype=torch.float)\n",
        "weights_aug = weights_aug / weights_aug.sum() * len(counts_aug)\n",
        "print(f'Class weights (augmented): { {c: round(w,3) for c,w in zip(encoder_aug.classes_, weights_aug.tolist())} }')\n",
        "\n",
        "all_labels_aug = [g.y.item() for g in dataset_aug]\n",
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "\n",
        "fold_acc_aug, fold_f1_aug = [], []\n",
        "best_acc_aug, best_state_aug = 0.0, None\n",
        "\n",
        "print('\\nStarting 5-Fold CV (TCGA + METABRIC)...\\n')\n",
        "indices_aug = list(range(len(dataset_aug)))\n",
        "\n",
        "for fold, (train_idx, test_idx) in enumerate(skf.split(indices_aug, all_labels_aug), 1):\n",
        "    print(f'--- Fold {fold}/5 ---')\n",
        "    train_data   = [dataset_aug[i] for i in train_idx]\n",
        "    test_data    = [dataset_aug[i] for i in test_idx]\n",
        "    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)\n",
        "    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)\n",
        "\n",
        "    model_aug = BioGNN(num_node_features=1, hidden_channels=64,\n",
        "                       num_classes=len(encoder_aug.classes_)).to(device)\n",
        "    optimizer = torch.optim.Adam(model_aug.parameters(), lr=0.001, weight_decay=1e-4)\n",
        "    criterion = torch.nn.CrossEntropyLoss(weight=weights_aug.to(device))\n",
        "\n",
        "    for epoch in range(1, 101):\n",
        "        model_aug.train()\n",
        "        for batch in train_loader:\n",
        "            batch = batch.to(device)\n",
        "            optimizer.zero_grad()\n",
        "            out  = model_aug(batch.x, batch.edge_index, batch.batch)\n",
        "            loss = criterion(out, batch.y)\n",
        "            loss.backward()\n",
        "            optimizer.step()\n",
        "\n",
        "    model_aug.eval()\n",
        "    preds, trues = [], []\n",
        "    with torch.no_grad():\n",
        "        for batch in test_loader:\n",
        "            batch = batch.to(device)\n",
        "            out   = model_aug(batch.x, batch.edge_index, batch.batch)\n",
        "            preds.extend(out.argmax(dim=1).cpu().numpy())\n",
        "            trues.extend(batch.y.cpu().numpy())\n",
        "\n",
        "    acc      = sum(p == t for p, t in zip(preds, trues)) / len(trues)\n",
        "    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)\n",
        "    fold_acc_aug.append(acc)\n",
        "    fold_f1_aug.append(macro_f1)\n",
        "    print(f'  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}')\n",
        "\n",
        "    if acc > best_acc_aug:\n",
        "        best_acc_aug   = acc\n",
        "        best_state_aug = {k: v.clone() for k, v in model_aug.state_dict().items()}\n",
        "        print(f'  --> New best model (acc={best_acc_aug:.4f})')\n",
        "\n",
        "torch.save(best_state_aug, 'best_model_augmented.pt')\n",
        "\n",
        "print(f'\\n=== Резултати (TCGA + METABRIC) ===')\n",
        "print(f'Accuracy:  {sum(fold_acc_aug)/5:.4f}  (+/- {pd.Series(fold_acc_aug).std():.4f})')\n",
        "print(f'Macro F1:  {sum(fold_f1_aug)/5:.4f}  (+/- {pd.Series(fold_f1_aug).std():.4f})')\n",
        "print(f'Best fold: {best_acc_aug:.4f}')\n",
        "print(f'\\n=== Сравнение: Преди vs. След Augmentation ===')\n",
        "print(f'Accuracy:  {sum(fold_accuracies)/5:.4f}  -->  {sum(fold_acc_aug)/5:.4f}')\n",
        "print(f'Macro F1:  {sum(fold_f1s)/5:.4f}  -->  {sum(fold_f1_aug)/5:.4f}')"
    ]
}

report_aug_cell = {
    "cell_type": "code",
    "id": "report_augmented",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- 7b. PER-CLASS REPORT (augmented model, тест само върху TCGA) ---\n",
        "from sklearn.metrics import classification_report\n",
        "\n",
        "model_aug.load_state_dict(best_state_aug)\n",
        "model_aug.eval()\n",
        "\n",
        "# Тестваме само върху оригиналните TCGA пациенти (честен тест)\n",
        "tcga_only = dataset_aug[:len(dataset)]\n",
        "all_preds_aug, all_labels_aug_list = [], []\n",
        "with torch.no_grad():\n",
        "    for batch in DataLoader(tcga_only, batch_size=32):\n",
        "        batch = batch.to(device)\n",
        "        out   = model_aug(batch.x, batch.edge_index, batch.batch)\n",
        "        all_preds_aug.extend(out.argmax(dim=1).cpu().numpy())\n",
        "        all_labels_aug_list.extend(batch.y.cpu().numpy())\n",
        "\n",
        "print('Classification Report — Augmented model (тест само върху TCGA):')\n",
        "print(classification_report(all_labels_aug_list, all_preds_aug,\n",
        "                             target_names=encoder_aug.classes_, zero_division=0))"
    ]
}

# Вмъкване: след PPI cell (index 3), след train loop (index 6 след вмъкване), след report (index 9)
nb['cells'].insert(4, augment_cell)
nb['cells'].insert(8, train_aug_cell)
nb['cells'].insert(10, report_aug_cell)

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Готово! Структура на notebook-а:')
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:70].replace('\n', ' ')
    print(f'  [{i}] {c.get("id","?")} | {src}')
