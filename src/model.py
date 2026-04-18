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
    GATConv(1 → hidden, heads=4)         →  ReLU  →  BatchNorm
    GATConv(hidden*4 → hidden, heads=4)  →  ReLU  →  BatchNorm
    GATConv(hidden*4 → hidden, heads=1)  →  ReLU
    GlobalMeanPool  →  Dropout  →  Linear(hidden → hidden//2)  →  ReLU  →  Linear → num_classes

    Промени спрямо предишната версия
    ----------------------------------
    - hidden_channels: 64 → 128  (по-голям капацитет за 160 гена)
    - Добавен трети GAT слой      (по-дълбоко разпространение на информация в PPI мрежата)
    - BatchNorm след всеки слой   (стабилизира обучението)
    - Два Linear слоя вместо един (по-добра класификация)

    Параметри
    ----------
    num_node_features : int
        Брой характеристики на всеки ген (= 1, генната експресия стойност)
    hidden_channels : int
        Размер на скрития слой (препоръчително: 128)
    num_classes : int
        Брой BRCA подтипове (обикновено 5)
    """

    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        super(BioGNN, self).__init__()

        # GAT слой 1: всеки ген "гледа" съседите си с 4 attention heads
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, dropout=0.2)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels * 4)

        # GAT слой 2: по-дълбоко разпространение
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=0.2)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels * 4)

        # GAT слой 3: обобщава в един вектор
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1,
                             concat=False, dropout=0.2)

        # Двустепенна класификация
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Графови конволюции с нормализация
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 2. Global pooling: усредняваме всички генни вектори → един вектор за пациента
        x = global_mean_pool(x, batch)

        # 3. Класификация с dropout
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return x
