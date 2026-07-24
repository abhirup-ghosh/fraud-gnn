"""Model inference: known-txId (via feature store) and inductive (raw features) paths."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from fraud_gnn import config as cfg
from fraud_gnn.featurestore import get_subgraph
from fraud_gnn.model import FraudSAGE


class InferenceEngine:
    def __init__(self, model_path=None, metadata_path=None, device=None):
        model_path = model_path or cfg.MODEL_PATH
        metadata_path = metadata_path or cfg.METADATA_PATH

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.device = device or torch.device("cpu")
        self.model = FraudSAGE(
            in_dim=self.metadata["in_dim"],
            hidden=self.metadata["hidden"],
            dropout=self.metadata["dropout"],
            aggr=self.metadata["aggr"],
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.threshold = self.metadata["f1_optimal_threshold"]
        self.model_version = self.metadata["model_version"]

    def _forward_center_proba(self, x: np.ndarray, edge_index: np.ndarray, center_idx: int) -> float:
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x_t, edge_index_t)
            proba = F.softmax(logits, dim=1)[:, 1]
        return float(proba[center_idx].item())

    def score_known(self, tx_id: int, feature_client=None) -> float:
        x, edge_index, center_idx = get_subgraph(tx_id, hops=1, client=feature_client)
        return self._forward_center_proba(x, edge_index, center_idx)

    def score_inductive(self, features: list[float], neighbor_features: list[list[float]] | None) -> float:
        neighbor_features = neighbor_features or []
        n_neighbors = len(neighbor_features)
        x = np.array([features] + list(neighbor_features), dtype=np.float32)

        if n_neighbors == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            src = [0] * n_neighbors + list(range(1, n_neighbors + 1))
            dst = list(range(1, n_neighbors + 1)) + [0] * n_neighbors
            edge_index = np.array([src, dst], dtype=np.int64)

        return self._forward_center_proba(x, edge_index, center_idx=0)
