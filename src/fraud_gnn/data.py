"""Loads the Elliptic CSVs into a PyG Data object with temporal train/val/test masks."""
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from fraud_gnn import config as cfg


def _stratified_val_split(y: torch.Tensor, train_idx: np.ndarray, val_fraction: float, seed: int):
    """Randomly carve a stratified val slice out of train_idx (not temporal)."""
    rng = np.random.default_rng(seed)
    val_parts = []
    for cls in (0, 1):
        cls_idx = train_idx[y[train_idx].numpy() == cls]
        n_val = int(round(len(cls_idx) * val_fraction))
        chosen = rng.choice(cls_idx, size=n_val, replace=False)
        val_parts.append(chosen)
    val_idx = np.concatenate(val_parts)
    remaining_train_idx = np.setdiff1d(train_idx, val_idx, assume_unique=False)
    return remaining_train_idx, val_idx


def load_elliptic(data_dir=None) -> tuple[Data, dict[int, int]]:
    data_dir = data_dir or cfg.DATA_DIR
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    feat = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)

    n_feat_cols = feat.shape[1] - 2
    feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, n_feat_cols + 1)]

    txid_to_idx = {int(tx): i for i, tx in enumerate(feat["txId"])}

    feature_cols = [f"feat_{i}" for i in range(1, n_feat_cols + 1)]
    x = torch.tensor(feat[feature_cols].to_numpy(dtype=np.float32))
    time_step = torch.tensor(feat["time_step"].to_numpy(dtype=np.int64))

    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes = classes.set_index("txId").reindex(feat["txId"])
    y_np = classes["class"].map(label_map).to_numpy(dtype=np.int64)
    y = torch.tensor(y_np)

    src = edges["txId1"].map(txid_to_idx).to_numpy(dtype=np.int64)
    dst = edges["txId2"].map(txid_to_idx).to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst]))
    edge_index = to_undirected(edge_index)

    labelled = y != -1
    labelled_idx = labelled.nonzero(as_tuple=True)[0].numpy()
    train_candidate_idx = labelled_idx[time_step[labelled_idx].numpy() <= cfg.TRAIN_MAX_STEP]
    test_idx = labelled_idx[time_step[labelled_idx].numpy() >= cfg.TEST_MIN_STEP]

    train_idx, val_idx = _stratified_val_split(y, train_candidate_idx, cfg.VAL_FRACTION, cfg.SEED)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        time_step=time_step,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.feature_columns = feature_cols
    return data, txid_to_idx


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """w_c = n_total / (2 * n_c), for classes {0: licit, 1: illicit}."""
    y_masked = y[mask]
    n_total = y_masked.numel()
    n_licit = (y_masked == 0).sum().item()
    n_illicit = (y_masked == 1).sum().item()
    w_licit = n_total / (2 * n_licit)
    w_illicit = n_total / (2 * n_illicit)
    return torch.tensor([w_licit, w_illicit], dtype=torch.float32)


def save_txid_to_idx(txid_to_idx: dict[int, int], path=None):
    path = path or cfg.TXID_TO_IDX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in txid_to_idx.items()}, f)
