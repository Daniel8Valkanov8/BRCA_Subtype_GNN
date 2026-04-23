"""
МОДУЛ: Graph Neural Network Architecture
ОПИСАНИЕ: BioGNN модел използващ Graph Attention Network v2 (GATv2) слоеве
за класификация на BRCA подтипове от генна експресия + PPI граф.
GATv2Conv приема edge_attr (combined_score от STRING), което позволява
на модела да взима предвид биологичната значимост на всяка протеин-протеинова
взаимодействие при изчисляване на attention weights.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class BioGNN(torch.nn.Module):
    """
    GATv2-базиран модел за класификация на тумор-подтипове с ребрени характеристики.

    Архитектура
    -----------
    GATv2Conv(1 → hidden, heads=4, edge_dim=1)         →  ReLU  →  BatchNorm
    GATv2Conv(hidden*4 → hidden, heads=4, edge_dim=1)  →  ReLU  →  BatchNorm
    GATv2Conv(hidden*4 → hidden, heads=1, edge_dim=1)  →  ReLU
    GlobalMeanPool  →  Dropout  →  Linear(hidden → hidden//2)  →  ReLU  →  Linear → num_classes

    Промени спрямо предишната версия
    ----------------------------------
    - GATConv → GATv2Conv: динамично attention (по-мощно от статичното в GATConv)
    - edge_dim=1: моделът получава combined_score от STRING като ребрена характеристика
      → attention weights вземат предвид биологичната сила на всяко PPI взаимодействие

    Параметри
    ----------
    num_node_features : int
        Брой характеристики на всеки ген (= 1, генната експресия стойност)
    hidden_channels : int
        Размер на скрития слой (препоръчително: 128)
    num_classes : int
        Брой BRCA подтипове (обикновено 5)
    edge_dim : int
        Размерност на ребрените характеристики (= 1 за normalized combined_score)
    """

    def __init__(self, num_node_features: int, hidden_channels: int,
                 num_classes: int, edge_dim: int = 1):
        super(BioGNN, self).__init__()

        self.conv1 = GATv2Conv(num_node_features, hidden_channels,
                               heads=4, dropout=0.2, edge_dim=edge_dim)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels * 4)

        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels,
                               heads=4, dropout=0.2, edge_dim=edge_dim)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels * 4)

        self.conv3 = GATv2Conv(hidden_channels * 4, hidden_channels,
                               heads=1, concat=False, dropout=0.2, edge_dim=edge_dim)

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None, return_attention=False):
        attn_weights = {}

        if return_attention:
            x, (ei1, aw1) = self.conv1(x, edge_index, edge_attr=edge_attr,
                                        return_attention_weights=True)
            attn_weights['layer1'] = (ei1.cpu(), aw1.cpu())
        else:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(self.bn1(x))

        if return_attention:
            x, (ei2, aw2) = self.conv2(x, edge_index, edge_attr=edge_attr,
                                        return_attention_weights=True)
            attn_weights['layer2'] = (ei2.cpu(), aw2.cpu())
        else:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(self.bn2(x))

        if return_attention:
            x, (ei3, aw3) = self.conv3(x, edge_index, edge_attr=edge_attr,
                                        return_attention_weights=True)
            attn_weights['layer3'] = (ei3.cpu(), aw3.cpu())
        else:
            x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)

        if return_attention:
            return x, attn_weights
        return x
