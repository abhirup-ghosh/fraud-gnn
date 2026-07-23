"""RandomForest baseline on the temporal split — the bar the GNN must beat."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_gnn import config as cfg
from fraud_gnn.data import load_elliptic


def run_baseline():
    data, _ = load_elliptic()

    X = data.x.numpy()
    y = data.y.numpy()

    train_mask = (data.train_mask | data.val_mask).numpy()  # RF has no early-stopping val need
    test_mask = data.test_mask.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    clf = RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        class_weight="balanced",
        random_state=cfg.SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print(f"train n={len(y_train)}  test n={len(y_test)}")
    print(f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}  "
          f"Precision={precision:.4f}  Recall={recall:.4f}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":
    run_baseline()
