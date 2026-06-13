"""
МОДУЛ: Подобрена GNN архитектура (BioGNNv3)
ОПИСАНИЕ: Адресира диагностицираните слабости на оригиналния BioGNN, заради
които той изоставаше драстично от прости baselines (RF/LogReg ~0.90 vs GNN 0.69).

Диагностицирани root causes и корекции
---------------------------------------
1. global_mean_pool заличаваше per-ген сигнала (усредняване на 196 nodes).
   → Корекция: dual readout [global_mean_pool ‖ global_max_pool] —
     max-pool запазва най-силните (дискриминативни) генни сигнали.
2. Over-smoothing от 3 GATv2 слоя.
   → Корекция: 2 слоя + JumpingKnowledge (конкат на изходите от всеки слой),
     така класификаторът вижда и по-ранни, по-малко изгладени представяния.
3. Слаб информационен поток към класификатора.
   → Корекция: по-широк readout (JK × dual-pool) + dropout регуляризация.

Запазва интерфейса на BioGNN: forward(x, edge_index, batch, edge_attr, return_attention).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class BioGNNv3(torch.nn.Module):
    """
    GATv2-базиран модел с JumpingKnowledge + dual readout.

    Архитектура
    -----------
    GATv2Conv(1 → hidden, heads=H)            → ReLU → BatchNorm   = h1  (dim hidden*H)
    GATv2Conv(hidden*H → hidden, heads=1)     → ReLU → BatchNorm   = h2  (dim hidden)
    JK: concat([h1, h2])                                            = h   (dim hidden*H + hidden)
    Readout: concat([mean_pool(h), max_pool(h)])                    = r   (dim 2*(hidden*H+hidden))
    Dropout → Linear(r → hidden) → ReLU → Dropout → Linear → classes
    """

    def __init__(self, num_node_features: int, hidden_channels: int,
                 num_classes: int, edge_dim: int = 1,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATv2Conv(num_node_features, hidden_channels,
                               heads=heads, dropout=0.2, edge_dim=edge_dim)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels,
                               heads=1, concat=False, dropout=0.2, edge_dim=edge_dim)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels)

        # JumpingKnowledge dim = изход conv1 (hidden*heads) + изход conv2 (hidden)
        jk_dim      = hidden_channels * heads + hidden_channels
        # Dual readout удвоява размерността (mean ‖ max)
        readout_dim = 2 * jk_dim

        self.lin1 = torch.nn.Linear(readout_dim, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None, return_attention=False):
        attn_weights = {}

        # ── Слой 1 ──
        if return_attention:
            h1, (ei1, aw1) = self.conv1(x, edge_index, edge_attr=edge_attr,
                                        return_attention_weights=True)
            attn_weights['layer1'] = (ei1.cpu(), aw1.cpu())
        else:
            h1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        h1 = F.relu(self.bn1(h1))

        # ── Слой 2 ──
        if return_attention:
            h2, (ei2, aw2) = self.conv2(h1, edge_index, edge_attr=edge_attr,
                                        return_attention_weights=True)
            attn_weights['layer2'] = (ei2.cpu(), aw2.cpu())
        else:
            h2 = self.conv2(h1, edge_index, edge_attr=edge_attr)
        h2 = F.relu(self.bn2(h2))

        # ── JumpingKnowledge: конкат на node-репрезентациите от двата слоя ──
        h = torch.cat([h1, h2], dim=1)

        # ── Dual readout: mean ‖ max ──
        h_mean = global_mean_pool(h, batch)
        h_max  = global_max_pool(h, batch)
        h = torch.cat([h_mean, h_max], dim=1)

        # ── Класификатор ──
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.lin2(h)

        if return_attention:
            return out, attn_weights
        return out
