import pandas as pd
import networkx as nx
from pyvis.network import Network

df = pd.read_csv("data/transactions.csv")[:10000]

G = nx.Graph()

for _, row in df.iterrows():

    customer = f"C_{row.customer_id}"
    merchant = f"M_{row.merchant_id}"
    device = f"D_{row.device_id}"

    fraud = row.is_fraud

    G.add_node(
        customer,
        label=customer,
        group="customer"
    )

    G.add_node(
        merchant,
        label=merchant,
        group="merchant"
    )

    G.add_node(
        device,
        label=device,
        group="device"
    )

    G.add_edge(customer, merchant)
    G.add_edge(customer, device)

net = Network(
    height="900px",
    width="100%",
    bgcolor="#222222",
    font_color="white"
)

net.from_nx(G)

net.show_buttons()

net.show("fraud_graph.html")