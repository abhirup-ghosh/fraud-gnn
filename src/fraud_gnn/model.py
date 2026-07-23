"""2-layer GraphSAGE node classifier for illicit-transaction detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from fraud_gnn import config as cfg


class FraudSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int = cfg.N_FEATURES,
        hidden: int = cfg.HIDDEN,
        out_dim: int = cfg.OUT_DIM,
        dropout: float = cfg.DROPOUT,
        aggr: str = cfg.AGGR,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr=aggr)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = SAGEConv(hidden, hidden, aggr=aggr)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.out(h)
