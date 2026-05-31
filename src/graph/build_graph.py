import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_graph(df):

    data = HeteroData()

    num_customers = df.customer_id.max() + 1
    num_merchants = df.merchant_id.max() + 1
    num_devices = df.device_id.max() + 1

    data["customer"].num_nodes = num_customers
    data["merchant"].num_nodes = num_merchants
    data["device"].num_nodes = num_devices

    edges_cm = df[["customer_id", "merchant_id"]].values.T
    edges_cd = df[["customer_id", "device_id"]].values.T

    data["customer", "tx_to", "merchant"].edge_index = torch.tensor(edges_cm)
    data["customer", "uses", "device"].edge_index = torch.tensor(edges_cd)

    return data


if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")
    graph = build_graph(df)
    torch.save(graph, "data/graph.pt")