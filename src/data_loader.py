"""
МОДУЛ: Data Loader & Preprocessing
ОПИСАНИЕ: Зарежда клиничните данни и генната експресия от cBioPortal,
напасва пациентите между двете бази и ги трансформира в графични обекти
за PyTorch Geometric. Поддържа edge features (combined_score от STRING).
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
    clinical_df = pd.read_csv(clinical_path, sep='\t', comment='#')
    clinical_df = clinical_df.dropna(subset=['Subtype'])

    exp_df = pd.read_csv(expression_path, sep='\t').set_index('Hugo_Symbol')
    if 'Entrez_Gene_Id' in exp_df.columns:
        exp_df = exp_df.drop(columns=['Entrez_Gene_Id'])

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
    Ако PPI DataFrame-ът съдържа колона 'combined_score', тя се нормализира
    до [0,1] и се прикача като edge_attr (ребрена характеристика) за GATv2Conv.

    Параметри
    ----------
    exp_df : pd.DataFrame
        Матрица (гени x пациенти)
    ppi_df : pd.DataFrame
        PPI мрежа с колони 'node1', 'node2' и опционално 'combined_score'
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

    y_encoded = le.fit_transform(clinical_df['Subtype'])
    print(f"Classes: {list(le.classes_)}")

    gene_map = {name: i for i, name in enumerate(exp_df.index)}

    has_score = 'combined_score' in ppi_df.columns

    edge_index_list = []
    edge_attr_list  = []

    for _, row in ppi_df.iterrows():
        if row['node1'] in gene_map and row['node2'] in gene_map:
            i, j = gene_map[row['node1']], gene_map[row['node2']]
            # STRING combined_score е в диапазон 0–1000; нормализираме до [0,1]
            score = float(row['combined_score']) / 1000.0 if has_score else 1.0
            edge_index_list += [[i, j], [j, i]]
            edge_attr_list  += [[score], [score]]

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
    print(f"Graph edges: {edge_index.shape[1]}  |  edge_attr: {edge_attr.shape}")

    for idx, patient in enumerate(exp_df.columns):
        x = torch.tensor(exp_df[patient].values, dtype=torch.float).view(-1, 1)
        y = torch.tensor([y_encoded[idx]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(graph)

    return data_list, le
