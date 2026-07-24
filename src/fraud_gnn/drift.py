"""PSI-based concept-drift monitor, tracking the top Phase-1 EDA features against a
frozen reference distribution computed from training data (see train.py)."""
from collections import deque

import numpy as np
import pandas as pd

from fraud_gnn import config as cfg

_DECILE_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def compute_psi(reference_edges: np.ndarray, values: np.ndarray, eps: float = 1e-4) -> float:
    """PSI of `values` against a reference decile distribution (10 equal-mass bins).

    `reference_edges` are the 9 internal decile boundaries (q10..q90) of the reference
    distribution; the outer bins extend to +/-inf so any incoming value — regardless of
    scale — always falls into a bin. `eps` keeps every term finite even for empty bins.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    n_bins = len(reference_edges) + 1
    if n == 0:
        return 0.0
    bin_edges = np.concatenate([[-np.inf], reference_edges, [np.inf]])
    counts, _ = np.histogram(values, bins=bin_edges)
    actual = counts / n
    expected = np.full(n_bins, 1.0 / n_bins)
    psi = np.sum((actual - expected) * np.log((actual + eps) / (expected + eps)))
    return float(psi)


class DriftMonitor:
    """Maintains a sliding buffer of incoming feature values and reports PSI-based drift."""

    def __init__(
        self,
        reference_stats_path=None,
        top_features=None,
        buffer_size: int = None,
        window: int = None,
    ):
        reference_stats_path = reference_stats_path or cfg.REFERENCE_STATS_PATH
        self.top_features = list(top_features or cfg.TOP_DRIFT_FEATURES)
        self.buffer_size = buffer_size or cfg.DRIFT_BUFFER
        self.window = window or cfg.DRIFT_WINDOW

        ref = pd.read_parquet(reference_stats_path).set_index("feature")
        self._edges = {
            feat: ref.loc[feat, [f"q{d}" for d in _DECILE_LEVELS]].to_numpy(dtype=float)
            for feat in self.top_features
            if feat in ref.index
        }
        self._buffers = {feat: deque(maxlen=self.buffer_size) for feat in self._edges}
        self._count = 0

        self.psi_by_feature: dict[str, float] = {feat: 0.0 for feat in self._edges}
        self.drift_score: float = 0.0

    def add(self, features: dict) -> None:
        """Record one incoming observation (dict of feature_name -> value)."""
        for feat in self._edges:
            if feat in features:
                self._buffers[feat].append(float(features[feat]))
        self._count += 1
        if self._count % self.window == 0:
            self.recompute()

    def recompute(self) -> float:
        """Force a PSI recomputation from the current buffer state."""
        psis = {}
        for feat, edges in self._edges.items():
            vals = np.array(self._buffers[feat])
            psis[feat] = compute_psi(edges, vals) if len(vals) else 0.0
        self.psi_by_feature = psis
        self.drift_score = float(np.mean(list(psis.values()))) if psis else 0.0
        return self.drift_score

    def status(self) -> str:
        if self.drift_score > cfg.DRIFT_PSI_ALERT:
            return "significant"
        if self.drift_score > cfg.DRIFT_PSI_WARN:
            return "moderate"
        return "stable"
