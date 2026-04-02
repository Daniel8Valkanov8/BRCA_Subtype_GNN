"""
МОДУЛ: Graph Neural Network Architecture
ОПИСАНИЕ: BioGNN модел използващ Graph Attention Network (GAT) слоеве
за класификация на BRCA подтипове от генна експресия + PPI граф.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class BioGNN(torch.nn.Module):
    """
    GAT-базиран модел за класификация на тумор-подтипове.

    Архитектура
    -----------
    GATConv(1 → hidden, heads=4)  →  ReLU
    GATConv(hidden*4 → hidden, heads=1)  →  ReLU
    GlobalMeanPool  →  Dropout  →  Linear(hidden → num_classes)

    Параметри
    ----------
    num_node_features : int
        Брой характеристики на всеки ген (= 1, генната експресия стойност)
    hidden_channels : int
        Размер на скрития слой (препоръчително: 64 или 128)
    num_classes : int
        Брой BRCA подтипове (обикновено 5)
    """

    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        super(BioGNN, self).__init__()

        # GAT слой 1: всеки ген "гледа" съседите си с 4 attention heads
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, dropout=0.2)

        # GAT слой 2: обобщава информацията в един вектор
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1,
                             concat=False, dropout=0.2)

        # Финална класификация
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Графови конволюции
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 2. Global pooling: усредняваме всички генни векторе → един вектор за пациента
        x = global_mean_pool(x, batch)

        # 3. Регуляризация и класификация
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
