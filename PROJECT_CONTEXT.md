# Дипломен проект: GNN модел за класификация на подтипове рак на гърдата

> **Версия на документа:** 2026-05-12  
> **Автор на анализа:** Claude Code (Anthropic Sonnet 4.6)  
> **Базиран на:** пълно четене на всички файлове в проектния директорий

---

## 1. Обзор

**Научен въпрос:** Може ли Graph Neural Network, конструирана върху протеин-протеинова взаимодействия (PPI) мрежа от STRING, да класифицира подтипове на рак на гърдата по-добре от стандартни подходи, използвайки генна експресия от TCGA като node features?

**Идея:** Всеки пациент се представя като **граф**, в който:
- **Върховете (nodes)** са гени (160 броя), с feature = mRNA z-score на пациента
- **Ребрата (edges)** са PPI взаимодействия от STRING, с тегло (edge feature) = `combined_score`
- **Лейбълът** е един от 5 BRCA молекулярни подтипа

Чрез многослойна Graph Attention мрежа (GATv2) моделът се учи кои протеин-протеинови взаимодействия са диагностично значими за всеки подтип.

**Целева задача:** 5-класова класификация — BRCA_Basal, BRCA_Her2, BRCA_LumA, BRCA_LumB, BRCA_Normal.

---

## 2. Източници на данни

### 2.1 TCGA (The Cancer Genome Atlas)

| Параметър | Стойност |
|---|---|
| Изследване | `brca_tcga_pan_can_atlas_2018` |
| Клинични данни | `brca_tcga_pan_can_atlas_2018_clinical_data.tsv` (572.9 KB) |
| Брой пациенти (с Subtype) | 981 |
| Тип данни | mRNA z-scores (RSEM нормализирани, от cBioPortal API) |
| Генни панели | 27 гена (оригинал) → 160 от 174 (текуща версия) |
| Изтеглени с | `fetch_expanded_genes.py` → `data/tcga_expression_174genes.csv` |

**Разпределение на подтиповете** (изчислено от `main.ipynb`, cell `27505087`):

| Подтип | Пациенти | Дял |
|---|---|---|
| BRCA_LumA | 499 | 50.9% |
| BRCA_LumB | 197 | 20.1% |
| BRCA_Basal | 171 | 17.4% |
| BRCA_Her2 | 78 | 7.9% |
| BRCA_Normal | 36 | 3.7% |

Дисбалансът е значителен — LumA е 13.9× по-чест от Normal. Адресира се с class weights + SMOTE.

**Нормализация:** Per-patient z-score (`train_tcga_only.py`, ред 41):
```python
tcga_expr = tcga_expr.apply(lambda col: (col - col.mean()) / (col.std() + 1e-8), axis=0)
```
Тази нормализация стандартизира стойностите на всеки пациент поотделно (axis=0 = по колони = по пациенти), т.е. прави се z-score на всеки пациент спрямо неговите собствени гени.

### 2.2 STRING (Protein-Protein Interactions)

| Параметър | Стойност |
|---|---|
| API endpoint | `https://string-db.org/api/tsv/network` |
| Вид | 9606 (Homo sapiens) |
| Confidence threshold | ≥ 400 (medium confidence, скала 0–1000) |
| Мрежа за 27 гена | `data/string_interactions_short.tsv` — 312 ребра |
| Мрежа за 160 гена | `data/string_ppi_160genes.tsv` — ~600+ ребра |
| Нормализация на теглото | `combined_score / 1000.0` → диапазон [0, 1] |

**Внимание:** `fetch_expanded_genes.py` (ред 71–85) изтегля мрежа и я записва като `string_ppi_174genes.tsv` (не `string_ppi_160genes.tsv`). Отделен скрипт `fetch_ppi_160genes.py` изтегля само 160-те гена с покритие в PPI.

### 2.3 METABRIC (планирано, не е завършено)

- Набор от ~1756 пациенти от cBioPortal, изследван в `apitests/TestMETABRICApi.py`
- `train_expanded.py` и notebook-клетки очакват файлове `data/metabric_expression_160genes.csv` и `data/metabric_clinical.csv` — **тези файлове НЕ СЪЩЕСТВУВАТ** в директорията
- Целта е да се разшири до ~2737 пациенти (981 TCGA + 1756 METABRIC)

### 2.4 Обработка и артефакти

```
data/
├── brca_tcga_pan_can_atlas_2018_clinical_data.tsv  ← клинични данни
├── tcga_expression_174genes.csv                    ← генна експресия 174 гена (от API)
├── mRNA_expression_fixed.txt                       ← оригинален 27-генен файл
├── string_interactions_short.tsv                   ← PPI за 27 гена
├── string_ppi_160genes.tsv                         ← PPI за 160 гена
├── all_174genes_list.tsv                           ← списък на 174 гена
└── fix_data.py                                     ← почиства mRNA_expression_fixed.txt
```

---

## 3. Архитектура на модела

### 3.1 BioGNN — текуща версия (`src/model.py`)

**Тип:** Graph Attention Network v2 (GATv2Conv) с edge features

```
Вход: x (N × 1) — генна експресия, edge_index, edge_attr (E × 1) — STRING score

Слой 1: GATv2Conv(1 → 128, heads=4, dropout=0.2, edge_dim=1)
         → BatchNorm1d(512)  →  ReLU
         Изход: (N × 512)

Слой 2: GATv2Conv(512 → 128, heads=4, dropout=0.2, edge_dim=1)
         → BatchNorm1d(512)  →  ReLU
         Изход: (N × 512)

Слой 3: GATv2Conv(512 → 128, heads=1, concat=False, dropout=0.2, edge_dim=1)
         → ReLU
         Изход: (N × 128)

Readout: global_mean_pool  →  (1 × 128) — граф-ниво представяне

Класификатор:
  Dropout(0.4)
  Linear(128 → 64)  →  ReLU
  Dropout(0.2)
  Linear(64 → 5)    →  logits

Краен изход: 5 класа (BRCA подтипове)
```

**Параметри:** `src/model.py`
- `num_node_features = 1` (само генна експресия)
- `hidden_channels = 128` (`train_tcga_only.py`, ред 21)
- `num_classes = 5`
- `edge_dim = 1`

**Ключова разлика спрямо GATConv:** GATv2Conv използва динамично attention (разл. от GATConv, което е статично), и поддържа `edge_dim` — стойността `combined_score` от STRING влиза директно в изчислението на attention weights.

### 3.2 Стара версия — notebook и backend (`main.ipynb`, `ui/backend/main.py`)

Оригиналният notebook (`main.ipynb`, cell `f1955af2`) и backend-ът на UI (`ui/backend/main.py`, редове 33–65) използват **GATConv** (не GATv2Conv), **без edge features**, с `hidden_channels=64`. 

**КРИТИЧНА НЕСЪОТВЕТСТВИЕ:** `ui/backend/main.py` дефинира локален клас `BioGNN` с `GATConv` и зарежда `best_model_tcga_160genes.pt`. Ако тоз checkpoint е записан от `train_tcga_only.py` (GATv2Conv архитектура), `model.load_state_dict(state)` ще ХВЪРЛИ ГРЕШКА при стартиране на сървъра — ключовете в state dict-а не съответстват.

---

## 4. Training pipeline

### 4.1 Текущ тренировъчен скрипт — `train_tcga_only.py`

| Параметър | Стойност |
|---|---|
| EPOCHS | 150 |
| BATCH_SIZE | 16 |
| LR | 0.0005 |
| WEIGHT_DECAY | 1e-4 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss(weight=class_weights) |
| Scheduler | ReduceLROnPlateau(patience=10, factor=0.5) |
| Early stopping | PATIENCE=20 епохи без подобрение |
| Валидация | 5-Fold Stratified K-Fold (random_state=42) |
| Метрики | Accuracy, Macro F1 |

**Class weights** (обратно-пропорционални):
- Изчисляват се от оригиналното разпределение преди SMOTE
- Нормализирани спрямо брой класове: `weights = (1/counts) / sum(1/counts) * 5`

**SMOTE (балансиране):**
- Прилага се *само* вътре в обучителния fold (предотвратява data leakage в тест набора)
- `k_neighbors=5` — работи дори с малкия BRCA_Normal клас (~29 обучителни проби)
- Синтетичните проби се инжектират като нови PyG `Data` обекти с базов `edge_index` и `edge_attr`

**Best model логика:** Запазва се state dict-ът с най-висока val accuracy в fold-а; след CV — глобалния най-добър.

**TensorBoard логове:** `runs/tcga_v2_fold_{1..5}/` — Loss/train, Accuracy/val, F1_macro/val, LR

### 4.2 Стар notebook pipeline — `main.ipynb`

| Параметър | Стойност |
|---|---|
| Гени | 27 |
| Epochs | 100 |
| hidden_channels | 64 |
| LR | 0.001 |
| Batch size | 16 |
| CV | 5-Fold StratifiedKFold |
| SMOTE | Не |
| Scheduler | Не |
| Early stopping | Не |

### 4.3 Разширен скрипт — `train_expanded.py` (незавършен)

Цели TCGA + METABRIC обучение с 160 гена, но съдържа множество проблеми (вж. Секция 7).

---

## 5. Структура на проекта

```
Master_Thesis_GNN/
│
├── src/                         — основен Python модул
│   ├── __init__.py              — (празен)
│   ├── data_loader.py           — load_and_align_data() + build_graph_dataset()
│   └── model.py                 — BioGNN клас (GATv2Conv)
│
├── data/                        — данни
│   ├── brca_tcga_pan_can_atlas_2018_clinical_data.tsv
│   ├── tcga_expression_174genes.csv   ← 174 гена × ~1000 пациента
│   ├── mRNA_expression_fixed.txt      ← 27 гена × 981 пациента (оригинал)
│   ├── string_interactions_short.tsv ← PPI за 27 гена
│   ├── string_ppi_160genes.tsv        ← PPI за 160 гена
│   ├── all_174genes_list.tsv          ← списък на 174 гена
│   └── fix_data.py                    ← еднократно почистване на данните
│
├── apitests/                    — изследователски скриптове (не влизат в pipeline-а)
│   ├── test_gdc_api.py          — тест на cBioPortal API
│   ├── test_string_api.py       — тест на STRING-DB API + hub genes анализ
│   └── TestMETABRICApi.py       — изследване на METABRIC данни
│
├── ui/
│   ├── backend/
│   │   └── main.py              — FastAPI backend (5 endpoints: /health, /predict, /discoveries, /statistics)
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx          — 2 таба: Предсказване, Биологични закономерности
│       │   └── components/
│       │       ├── PredictionTab.jsx   — upload на CSV → резултат
│       │       └── StatisticsPanel.jsx — статистики за двойка гени
│       └── package.json         — React + Vite + Tailwind
│
├── notebooks/                   — (празна, само .gitkeep)
├── runs/                        — TensorBoard логове от тренировки
│
├── main.ipynb                   — оригинален notebook (27 гена, пълен pipeline)
├── train_tcga_only.py           — основен тренировъчен скрипт (160 гена, GATv2)
├── train_expanded.py            — разширен скрипт (TCGA+METABRIC, незавършен)
├── fetch_expanded_genes.py      — изтегля 174 гена от cBioPortal + PPI от STRING
├── fetch_ppi_160genes.py        — изтегля PPI за 160 гена от STRING
├── update_notebook.py           — добавя METABRIC клетки към main.ipynb
│
├── best_model_tcga_160genes.pt  — запазен checkpoint (вероятно GATConv, Acc=0.5851)
├── Last_model_tcga_160genes(1).pt  — нов checkpoint (неследен от git)
├── gnn_latent_space.png         — t-SNE визуализация на latent space
├── patient_debug.json           — тестов пациент (TCGA-BH-A0HF)
├── requirements.txt             — зависимости (без версионно закрепване)
├── commands.txt                 — команди за стартиране на UI
└── venv/                        — Python virtual environment
```

---

## 6. Текущо състояние

### Работи и е тествано ✓
- **Данните** са заредени и изравнени: 981 пациента × 174 гена (TCGA), класове кодирани
- **PPI мрежата** е изградена: 160-ген граф с edge weights от STRING
- **PyG графовете** се конструират правилно: `build_graph_dataset()` работи
- **train_tcga_only.py** е написан и изпълнен: 5-Fold CV с GATv2Conv + SMOTE
- **TensorBoard логове** съществуват в `runs/`
- **UI** е имплементиран: FastAPI backend + React frontend, стартира на порт 5173/8000
- **main.ipynb** е изпълнен с 27-генен модел: Acc=0.5658, Macro F1=0.4591

### Имплементирано, но непроверено / неизпълнено
- **GATv2Conv upgrade** — `src/model.py` и `train_tcga_only.py` са обновени, но не е ясно дали `best_model_tcga_160genes.pt` е регенериран с новата архитектура или е от стария GATConv
- **UI backend** — `ui/backend/main.py` е маркиран като модифициран (`M`) в git, но все още ползва старата GATConv архитектура

### Не е започнато
- **METABRIC интеграция** — файловете `data/metabric_expression_160genes.csv` и `data/metabric_clinical.csv` не съществуват; `train_expanded.py` не може да се изпълни
- **Анализ на attention weights** — функционалността е в UI backend, но не е систематично анализирана
- **Hyperparameter tuning** — стойностите са избрани евристично
- **Сравнение с baseline** (Random Forest / Logistic Regression върху същите features)
- **Статистическа значимост** на резултатите

---

## 7. Известни проблеми и препятствия

### 7.1 КРИТИЧНО: Несъответствие на архитектурата в UI backend
**Файл:** `ui/backend/main.py`, редове 31–65  
**Проблем:** Backend-ът дефинира локален `BioGNN` клас с **GATConv** (без edge features, hidden=128) и зарежда `best_model_tcga_160genes.pt`. Ако checkpoint-ът е записан от `train_tcga_only.py` с **GATv2Conv**, `load_state_dict` ще хвърли `RuntimeError: Missing/Unexpected keys` при стартиране.  
**Статус:** `git status` показва `M ui/backend/main.py` — файлът е модифициран в последния commit, но все още ползва GATConv.  
**Решение:** Или актуализирай backend-а да ползва GATv2Conv с edge_attr, или изясни кой точно checkpoint трябва да зареди.

### 7.2 КРИТИЧНО: `train_expanded.py` извиква модела без `edge_attr`
**Файл:** `train_expanded.py`, редове 99 и 112 и 161  
```python
out = model(batch.x, batch.edge_index, batch.batch)  # ЛИПСВА edge_attr
```
**Проблем:** `BioGNN` е конструиран с `edge_dim=1` по подразбиране. GATv2Conv очаква `edge_attr` при forward pass. Без него ще хвърли грешка при изпълнение.  
**Решение:** Добави `edge_attr=batch.edge_attr` към всяко извикване.

### 7.3 КРИТИЧНО: Липсващи METABRIC файлове
**Файл:** `train_expanded.py`, редове 34–38  
```python
meta_expr = pd.read_csv('data/metabric_expression_160genes.csv', index_col=0)
meta_clin = pd.read_csv('data/metabric_clinical.csv', index_col=0)
```
**Проблем:** Тези файлове не съществуват. Скриптът е нефункционален.  
**Решение:** Трябва да се напише скрипт за изтегляне на METABRIC данни от cBioPortal (изследван в `TestMETABRICApi.py`, но не е довършен). Вероятно `apitests/TestMETABRICApi.py` е отправна точка.

### 7.4 ВАЖНО: t-SNE hook в `main.ipynb` е несъвместим с новия модел
**Файл:** `main.ipynb`, cell `w8ja9gxg82`  
```python
handle = model.lin.register_forward_hook(hook_fn)
```
**Проблем:** В новия `BioGNN` (`src/model.py`) линейните слоеве се казват `lin1` и `lin2`, не `lin`. Тази клетка ще хвърли `AttributeError: 'BioGNN' object has no attribute 'lin'`.  
**Решение:** Промени на `model.lin1.register_forward_hook(...)` или `model.lin2...`.

### 7.5 ВАЖНО: Несъответствие в именуването на METABRIC файловете
**В notebook** (`main.ipynb`, cell `metabric_augment`): очаква `data/metabric_expression.csv` (27 гена)  
**В `train_expanded.py`**: очаква `data/metabric_expression_160genes.csv` (160 гена)  
Двата файла са различни — ако METABRIC бъде интегриран, трябва да се реши кой формат се използва.

### 7.6 ВАЖНО: `train_expanded.py` не използва SMOTE
Разширеният скрипт няма SMOTE, за разлика от `train_tcga_only.py`. При смесен датасет (TCGA+METABRIC) разпределението на класовете може да е различно — дисбалансът може да е по-малко изявен, но не е гарантирано.

### 7.7 УМЕРЕНО: `train_expanded.py` няма early stopping
**Файл:** `train_expanded.py`, ред 93 — `for epoch in range(1, EPOCHS + 1):`  
Липсва PATIENCE логиката от `train_tcga_only.py`. Моделът ще тренира всички 150 епохи без механизъм за спиране при overfitting.

### 7.8 УМЕРЕНО: `train_expanded.py` записва финалните (не най-добрите) стойности за fold_accs
**Файл:** `train_expanded.py`, ред 134  
```python
fold_accs.append(val_acc)  # val_acc е от последната епоха
```
За разлика от `train_tcga_only.py` (ред 154), където се записва `best_fold_acc`. Финалните резултати ще са под-оптимални.

### 7.9 УМЕРЕНО: `requirements.txt` без версионно закрепване
**Файл:** `requirements.txt`  
Съдържа само имена на пакети без версии (`torch`, `torch_geometric` и т.н.). PyTorch и PyTorch Geometric имат строги версионни зависимости между себе си и CUDA. При пресъздаване на средата в различна система — висок риск от несъвместими версии.

### 7.10 ИНФОРМАЦИОННО: `fetch_expanded_genes.py` записва `string_ppi_174genes.tsv`
**Файл:** `fetch_expanded_genes.py`, ред 85  
Скриптът записва `data/string_ppi_174genes.tsv`, но `train_expanded.py` (ред 56) го зарежда именно с това име. Обаче `fetch_ppi_160genes.py` записва отделен `data/string_ppi_160genes.tsv`. Двата файла имат различно покритие.

---

## 8. Зависимости и среда

### Python и ML библиотеки

| Библиотека | Текуща вер. (от venv) | Роля |
|---|---|---|
| Python | ~3.x (venv в проекта) | — |
| pandas | 3.0.2 | данни |
| numpy | 2.4.4 | числени операции |
| torch | (не е фиксирана) | дълбоко учене |
| torch_geometric | (не е фиксирана) | GNN слоеве |
| scikit-learn | (не е фиксирана) | CV, метрики, LabelEncoder |
| imbalanced-learn | (не е фиксирана) | SMOTE |
| matplotlib / seaborn | (не е фиксирани) | визуализация |
| scipy | 1.17.1 | статистика (Pearson/Spearman в backend) |
| fastapi / uvicorn | (не са фиксирани) | UI backend |

### Frontend

| Технология | Версия |
|---|---|
| React | (от package.json) |
| Vite | (build tool) |
| Tailwind CSS | (styling) |

### Hardware

- Тренировката в `train_tcga_only.py` автоматично открива CUDA: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- `train_expanded.py` твърдо задава `device = torch.device('cpu')` (ред 64)
- Размерите на моделите (160 гена, GATv2, 128 hidden) са относително малки — CPU тренировката е осъществима, но бавна

### Стартиране на UI

```powershell
# Backend
cd "c:/Users/danie/Desktop/Master_Thesis_GNN/ui/backend"
../../venv/Scripts/python -m uvicorn main:app --reload

# Frontend (в отделен терминал)
cd "c:/Users/danie/Desktop/Master_Thesis_GNN/ui/frontend"
npm run dev
# → http://localhost:5173
```

---

## 9. TODO / Следващи стъпки (по приоритет)

### Критично (блокиращо)
1. **Изясни кой checkpoint е `best_model_tcga_160genes.pt`** — GATConv или GATv2Conv? Провери с `torch.load('best_model_tcga_160genes.pt').keys()`
2. **Синхронизирай UI backend** — `ui/backend/main.py` трябва да ползва същата архитектура като тренировъчния скрипт (GATv2Conv с edge_dim=1)
3. **Поправи `train_expanded.py`** — добави `edge_attr=batch.edge_attr` навсякъде

### Висок приоритет
4. **Изтегли METABRIC данни** за 160-те гена и ги форматирай като `data/metabric_expression_160genes.csv`
5. **Изпълни `train_tcga_only.py` с новата GATv2Conv архитектура** и провери дали резултатите се подобряват спрямо Acc=0.5851 (GATConv)
6. **Закрепи версиите в `requirements.txt`** — особено torch, torch_geometric, CUDA версия

### Среден приоритет
7. **Добави SMOTE и early stopping в `train_expanded.py`**
8. **Поправи t-SNE hook** в `main.ipynb` (cell `w8ja9gxg82`) — `model.lin` → `model.lin1`
9. **Baseline сравнение** — Random Forest / Logistic Regression върху същите 160-генни профили

### Нисък приоритет (дипломна завършеност)
10. Систематичен анализ на attention weights по подтипове
11. Добавяне на confusion matrix към тренировъчните резултати
12. Документация на резултатите с таблица за сравнение на всички конфигурации

---

## 10. Отворени въпроси

1. **Кой модел зарежда UI backend?** Коментарът на ред 29 в `ui/backend/main.py` казва `"GATConv — съответства на best_model_tcga_160genes.pt"`, но тренировъчният скрипт ползва GATv2Conv. Има ли два различни checkpoint файла — един от преди upgrade-а (GATConv) и един след него (GATv2Conv)? Какво е `Last_model_tcga_160genes(1).pt`?

2. **Изпълнен ли е `train_tcga_only.py` с новата GATv2Conv архитектура?** Ако да, какви са резултатите? Сравнението в ред 186 показва placeholder `Acc={sum(fold_accs)/5:.4f}` — т.е. не е hard-coded, а изчисляван при изпълнение. Има ли запазени резултати?

3. **Нормализацията `per-patient z-score` в `train_tcga_only.py` (ред 41)** — `axis=0` означава по колони (пациенти). Това е правилна per-patient нормализация. В `main.ipynb` обаче такава нормализация няма — изобщо. Умишлено ли е?

4. **`train_expanded.py` задава `device = torch.device('cpu')` твърдо** — защо? Дали CUDA е недостъпна за тази конфигурация?

5. **`train_expanded.py` ред 159: `dataset[:len(common_tcga)]`** — per-class report-ът се изчислява само върху TCGA пациентите от финалния dataset. Но `dataset` е комбиниран (TCGA+METABRIC). Ако обаче TCGA пациентите са на позиции 0..len(common_tcga)-1, това е коректно. Но при SMOTE или shuffle — дали редът се запазва?

6. **Как точно се изчисляват edge features в backend-а?** `ui/backend/main.py`, функция `load_resources()` (редове 93–98) изгражда `edge_index` без `edge_attr`. Дори ако GATConv архитектурата не ги ползва, при преминаване към GATv2Conv ще е нужно и да се запази `edge_attr`.

7. **Какъв е точният брой гени в тренировката?** `tcga_expression_174genes.csv` съдържа 174 гена, но `string_ppi_160genes.tsv` покрива само 160. `build_graph_dataset()` използва само гени с покритие в PPI (`if row['node1'] in gene_map and row['node2'] in gene_map`). Ефективно колко гена (nodes) има в графа?

8. **Subtype label encoding** — `LabelEncoder` сортира класовете по азбучен ред: `['BRCA_Basal', 'BRCA_Her2', 'BRCA_LumA', 'BRCA_LumB', 'BRCA_Normal']` = [0, 1, 2, 3, 4]. Това е консистентно между тренировка и backend (`CLASSES` в `main.py`, ред 74). Но при промяна на данните трябва внимателно — LabelEncoder не гарантира консистентен ред ако класовете се промените.

---

*Документът описва състоянието на проекта към 2026-05-12 и трябва да се актуализира след всяка значима промяна в архитектурата или данните.*
