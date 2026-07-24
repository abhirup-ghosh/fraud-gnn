"""Full-batch GraphSAGE training with early stopping on validation PR-AUC."""
import copy
import json
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from fraud_gnn import config as cfg
from fraud_gnn.data import compute_class_weights, load_elliptic, save_txid_to_idx
from fraud_gnn.model import FraudSAGE


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Training device.

    MPS is detected and reported, but full-batch training runs on CPU: empirically,
    MPS gives non-deterministic, unstable training trajectories for this graph (SAGEConv's
    scatter-reduce ops are non-deterministic on the MPS backend) — two runs varied from
    val PR-AUC 0.67 to 0.98, even with `torch.use_deterministic_algorithms(warn_only=True)`.
    CPU is bit-for-bit reproducible here and full-batch training on this graph size
    (203k nodes) still completes in under two minutes. See docs/plan.md §S4.
    Set FRAUD_GNN_FORCE_MPS_TRAIN=1 to override (non-deterministic; not recommended).
    """
    mps_available = torch.backends.mps.is_available()
    print(f"mps available: {mps_available}")
    if mps_available and os.environ.get("FRAUD_GNN_FORCE_MPS_TRAIN") == "1":
        return torch.device("mps")
    return torch.device("cpu")


def _model_version() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha
    except Exception:
        return time.strftime("%Y%m%dT%H%M%S")


def _f1_optimal_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    # precision_recall_curve returns one more point than thresholds; drop the last
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def _save_reference_stats(x: np.ndarray, feature_columns: list[str], path):
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for i, col in enumerate(feature_columns):
        vals = x[:, i]
        row = {"feature": col, "mean": float(vals.mean()), "std": float(vals.std())}
        qs = np.quantile(vals, quantile_levels)
        for level, q in zip(quantile_levels, qs):
            row[f"q{int(level * 100)}"] = float(q)
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def train():
    seed_everything(cfg.SEED)
    device = get_device()
    print(f"device: {device}")

    data, txid_to_idx = load_elliptic()
    data = data.to(device)
    class_weights = compute_class_weights(data.y, data.train_mask).to(device)

    model = FraudSAGE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_val_pr_auc = -1.0
    best_epoch = -1
    best_state = None
    epochs_since_improve = 0

    y_val = data.y[data.val_mask].detach().cpu().numpy()

    for epoch in range(cfg.EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask], weight=class_weights
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[data.val_mask]
            val_proba = F.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
        val_pr_auc = average_precision_score(y_val, val_proba)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 10 == 0 or epochs_since_improve == 0:
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  val_pr_auc={val_pr_auc:.4f}  "
                  f"best={best_val_pr_auc:.4f}@{best_epoch}")

        if epochs_since_improve >= cfg.PATIENCE:
            print(f"early stopping at epoch {epoch} (no improvement for {cfg.PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(data.x, data.edge_index)[data.val_mask]
        val_proba = F.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
    f1_threshold = _f1_optimal_threshold(y_val, val_proba)
    val_pred = (val_proba >= f1_threshold).astype(int)
    print(f"best val PR-AUC={best_val_pr_auc:.4f} @ epoch {best_epoch}  "
          f"f1_optimal_threshold={f1_threshold:.4f}  val_f1={f1_score(y_val, val_pred):.4f}")

    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    save_txid_to_idx(txid_to_idx)

    x_train_np = data.x[data.train_mask].detach().cpu().numpy()
    _save_reference_stats(x_train_np, data.feature_columns, cfg.REFERENCE_STATS_PATH)

    metadata = {
        "in_dim": cfg.N_FEATURES,
        "hidden": cfg.HIDDEN,
        "dropout": cfg.DROPOUT,
        "aggr": cfg.AGGR,
        "best_val_pr_auc": best_val_pr_auc,
        "best_epoch": best_epoch,
        "f1_optimal_threshold": f1_threshold,
        "feature_columns": data.feature_columns,
        "model_version": _model_version(),
    }
    with open(cfg.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved artefacts to {cfg.ARTIFACTS_DIR}")
    return model, data


if __name__ == "__main__":
    train()
