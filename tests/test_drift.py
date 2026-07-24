import numpy as np
import pandas as pd
import pytest

from fraud_gnn import config as cfg
from fraud_gnn.drift import DriftMonitor, compute_psi


@pytest.fixture(scope="module")
def reference():
    return pd.read_parquet(cfg.REFERENCE_STATS_PATH).set_index("feature")


@pytest.fixture()
def feature_and_edges(reference):
    feat = cfg.TOP_DRIFT_FEATURES[0]
    edges = reference.loc[feat, [f"q{d}" for d in [10, 20, 30, 40, 50, 60, 70, 80, 90]]].to_numpy(
        dtype=float
    )
    std = float(reference.loc[feat, "std"])
    return feat, edges, std


def _uniform_decile_values(edges: np.ndarray, n_per_bin: int = 50) -> np.ndarray:
    """Values placed exactly at the midpoint of each of the 10 equal-mass reference bins."""
    all_edges = np.concatenate([[edges[0] - abs(edges[0]) - 1], edges, [edges[-1] + abs(edges[-1]) + 1]])
    midpoints = (all_edges[:-1] + all_edges[1:]) / 2
    return np.repeat(midpoints, n_per_bin)


def test_psi_zero_for_matching_distribution(feature_and_edges):
    _, edges, _ = feature_and_edges
    values = _uniform_decile_values(edges)
    psi = compute_psi(edges, values)
    assert abs(psi) < 1e-6


def test_psi_large_for_shifted_distribution(feature_and_edges):
    _, edges, std = feature_and_edges
    values = _uniform_decile_values(edges) + 3 * std
    psi = compute_psi(edges, values)
    assert psi > 0.25


def test_drift_monitor_finite_and_no_raise_on_extreme_values():
    monitor = DriftMonitor()
    rng = np.random.default_rng(0)
    for _ in range(cfg.DRIFT_WINDOW + 5):
        obs = {feat: float(rng.normal()) for feat in monitor.top_features}
        # throw in an absurd out-of-scale value for one feature
        obs[monitor.top_features[0]] = 1e12
        monitor.add(obs)

    assert np.isfinite(monitor.drift_score)
    for psi in monitor.psi_by_feature.values():
        assert np.isfinite(psi)


def test_drift_monitor_status_thresholds():
    monitor = DriftMonitor()
    monitor.drift_score = 0.05
    assert monitor.status() == "stable"
    monitor.drift_score = 0.15
    assert monitor.status() == "moderate"
    monitor.drift_score = 0.30
    assert monitor.status() == "significant"
