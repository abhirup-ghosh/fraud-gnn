import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGNN(nn.Module):

    def __init__(self, in_dim=16):

        super().__init__()

        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 32)
        self.lin = nn.Linear(32, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return torch.sigmoid(self.lin(x))