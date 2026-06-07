import joblib
import torch
import numpy as np

from src.models.gnn import FraudGNN

# XGBoost
xgb_model = joblib.load("artifacts/xgb.pkl")

# GNN
gnn_model = FraudGNN()
gnn_model.load_state_dict(torch.load("artifacts/gnn.pt"))
gnn_model.eval()

def compute_score(tx):

    x = np.array([[
        tx.amount,
        tx.hour,
        tx.day
    ]])

    xgb_score = xgb_model.predict_proba(x)[:, 1][0]

    # placeholder graph embedding (v1 simplification)
    x_graph = torch.randn((1, 16))
    edge_index = torch.randint(0, 10, (2, 20))

    gnn_score = gnn_model(x_graph, edge_index).detach().numpy()[0][0]

    return 0.4 * xgb_score + 0.6 * gnn_score