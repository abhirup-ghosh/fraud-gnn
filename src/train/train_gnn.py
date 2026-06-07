import torch
from torch_geometric.loader import NeighborLoader
from src.models.gnn import FraudGNN


def train(graph):

    model = FraudGNN()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):

        optimizer.zero_grad()

        x = torch.randn((graph.num_nodes, 16))
        edge_index = graph["customer", "uses", "device"].edge_index

        out = model(x, edge_index)

        loss = out.mean()

        loss.backward()
        optimizer.step()

        print(epoch, loss.item())

    torch.save(model.state_dict(), "artifacts/gnn.pt")