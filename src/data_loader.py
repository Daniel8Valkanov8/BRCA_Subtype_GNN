"""
МОДУЛ: Data Loader & Preprocessing
ОПИСАНИЕ: Зарежда клиничните данни и генната експресия от cBioPortal,
напасва пациентите между двете бази и ги трансформира в графични обекти
за PyTorch Geometric.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder


def load_and_align_data(clinical_path: str, expression_path: str):
    """
    Напасва пациентите между клиничните данни и матрицата на генната експресия.

    Параметри
    ----------
    clinical_path : str
        Път до 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv'
    expression_path : str
        Път до 'data_mrna_seq_v2_rsem.txt' (изтеглен от cBioPortal)

    Връща
    ------
    exp_final : pd.DataFrame
        Матрица (гени x пациенти) само с общите пациенти
    clinical_final : pd.DataFrame
        Клинични данни индексирани по Sample ID за общите пациенти
    """
    # Клинични данни — пропускаме коментарите (#) в началото на файла
    clinical_df = pd.read_csv(clinical_path, sep='\t', comment='#')
    clinical_df = clinical_df.dropna(subset=['Subtype'])

    # Генна експресия — транспонираме така, че редовете да са гени
    exp_df = pd.read_csv(expression_path, sep='\t').set_index('Hugo_Symbol')
    if 'Entrez_Gene_Id' in exp_df.columns:
        exp_df = exp_df.drop(columns=['Entrez_Gene_Id'])

    # Намираме общите пациенти (intersection между двата файла)
    common_samples = list(set(exp_df.columns) & set(clinical_df['Sample ID']))
    print(f"Common patients: {len(common_samples)}")

    exp_final = exp_df[common_samples]
    clinical_final = clinical_df.set_index('Sample ID').loc[common_samples]

    return exp_final, clinical_final


def build_graph_dataset(exp_df: pd.DataFrame,
                        ppi_df: pd.DataFrame,
                        clinical_df: pd.DataFrame):
    """
    Превръща таблиците в списък от PyG Data обекти — по един граф на пациент.

    Параметри
    ----------
    exp_df : pd.DataFrame
        Матрица (гени x пациенти)
    ppi_df : pd.DataFrame
        PPI мрежа с колони 'node1' и 'node2' (STRING Database)
    clinical_df : pd.DataFrame
        Клинични данни с колона 'Subtype', индексирани по Sample ID

    Връща
    ------
    data_list : list[Data]
        Списък с PyG графове — един на пациент
    label_encoder : LabelEncoder
        Енкодерът, за да можем да декодираме предсказанията обратно
    """
    data_list = []
    le = LabelEncoder()

    # Подтиповете: BRCA_LumA, BRCA_LumB, BRCA_Her2, BRCA_Basal, BRCA_Normal → 0..4
    y_encoded = le.fit_transform(clinical_df['Subtype'])
    print(f"Classes: {list(le.classes_)}")

    # Речник: ген → индекс (нужен за изграждане на edge_index)
    gene_map = {name: i for i, name in enumerate(exp_df.index)}

    # Edge index — ребрата на графа от PPI мрежата
    edge_index_list = []
    for _, row in ppi_df.iterrows():
        if row['node1'] in gene_map and row['node2'] in gene_map:
            i, j = gene_map[row['node1']], gene_map[row['node2']]
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])  # графът е ненасочен

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    print(f"Graph edges: {edge_index.shape[1]}")

    # Един граф на пациент — характеристиките (x) се менят, топологията (edges) остава
    for i, patient in enumerate(exp_df.columns):
        x = torch.tensor(exp_df[patient].values, dtype=torch.float).view(-1, 1)
        y = torch.tensor([y_encoded[i]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(graph)

    return data_list, le
