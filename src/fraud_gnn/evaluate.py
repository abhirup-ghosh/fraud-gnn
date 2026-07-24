"""Evaluation on the temporal test set + per-time-step drift table (the acceptance gate)."""
import json

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_gnn import config as cfg
from fraud_gnn.data import load_elliptic
from fraud_gnn.model import FraudSAGE
from fraud_gnn.train import get_device


def _metrics_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def evaluate(model=None, data=None):
    device = get_device()

    if model is None or data is None:
        data, _ = load_elliptic()
        data = data.to(device)
        with open(cfg.METADATA_PATH) as f:
            metadata = json.load(f)
        model = FraudSAGE(
            in_dim=metadata["in_dim"],
            hidden=metadata["hidden"],
            dropout=metadata["dropout"],
            aggr=metadata["aggr"],
        ).to(device)
        model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    else:
        with open(cfg.METADATA_PATH) as f:
            metadata = json.load(f)

    threshold = metadata["f1_optimal_threshold"]
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        proba_all = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    y_test = data.y[data.test_mask].detach().cpu().numpy()
    proba_test = proba_all[data.test_mask.detach().cpu().numpy()]

    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)
    at_05 = _metrics_at_threshold(y_test, proba_test, 0.5)
    at_opt = _metrics_at_threshold(y_test, proba_test, threshold)

    print(f"test n={len(y_test)}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"  @0.5              F1={at_05['f1']:.4f}  P={at_05['precision']:.4f}  "
          f"R={at_05['recall']:.4f}")
    print(f"  @opt({threshold:.3f})     F1={at_opt['f1']:.4f}  P={at_opt['precision']:.4f}  "
          f"R={at_opt['recall']:.4f}")

    # Per-time-step table for steps 35-49
    time_step = data.time_step.detach().cpu().numpy()
    y_np = data.y.detach().cpu().numpy()
    test_mask_np = data.test_mask.detach().cpu().numpy()

    print(f"\n{'time_step':>10} {'n':>6} {'n_illicit':>10} {'roc_auc':>9} {'f1':>7}")
    per_step_rows = []
    for t in range(cfg.TEST_MIN_STEP, 50):
        step_mask = test_mask_np & (time_step == t)
        n = int(step_mask.sum())
        if n == 0:
            continue
        y_step = y_np[step_mask]
        n_illicit = int((y_step == 1).sum())
        if n_illicit == 0 or n_illicit == n:
            step_auc = float("nan")
        else:
            proba_step = proba_all[step_mask]
            step_auc = roc_auc_score(y_step, proba_step)
        proba_step = proba_all[step_mask]
        pred_step = (proba_step >= threshold).astype(int)
        step_f1 = f1_score(y_step, pred_step, zero_division=0)
        print(f"{t:>10} {n:>6} {n_illicit:>10} {step_auc:>9.4f} {step_f1:>7.4f}")
        per_step_rows.append(
            {"time_step": t, "n": n, "n_illicit": n_illicit, "roc_auc": step_auc, "f1": step_f1}
        )

    gate_passed = pr_auc >= 0.80 and at_opt["f1"] >= 0.80
    print(f"\nAcceptance gate (PR-AUC>=0.80 and F1>=0.80 at optimal threshold): "
          f"{'PASSED' if gate_passed else 'FAILED'}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_at_05": at_05["f1"],
        "f1_at_opt": at_opt["f1"],
        "precision_at_opt": at_opt["precision"],
        "recall_at_opt": at_opt["recall"],
        "per_step": per_step_rows,
        "gate_passed": gate_passed,
    }


if __name__ == "__main__":
    evaluate()
