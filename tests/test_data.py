import pytest
import torch

from fraud_gnn import config as cfg
from fraud_gnn.data import compute_class_weights, load_elliptic


@pytest.fixture(scope="module")
def loaded():
    return load_elliptic()


def test_shapes(loaded):
    data, _ = loaded
    assert data.x.shape == (cfg.N_NODES, cfg.N_FEATURES)
    assert data.y.shape == (cfg.N_NODES,)
    assert data.time_step.shape == (cfg.N_NODES,)


def test_edge_index_symmetric(loaded):
    data, _ = loaded
    edges = set(map(tuple, data.edge_index.t().tolist()))
    reversed_edges = {(b, a) for a, b in edges}
    assert reversed_edges <= edges


def test_masks_disjoint_and_no_unknown(loaded):
    data, _ = loaded
    train, val, test = data.train_mask, data.val_mask, data.test_mask
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    for mask in (train, val, test):
        assert (data.y[mask] != -1).all()


def test_train_val_covers_labelled_pre_split(loaded):
    data, _ = loaded
    labelled = data.y != -1
    expected = (labelled & (data.time_step <= cfg.TRAIN_MAX_STEP)).sum().item()
    actual = data.train_mask.sum().item() + data.val_mask.sum().item()
    assert actual == expected


def test_val_fraction_and_stratification(loaded):
    data, _ = loaded
    n_train = data.train_mask.sum().item()
    n_val = data.val_mask.sum().item()
    val_frac = n_val / (n_train + n_val)
    assert abs(val_frac - cfg.VAL_FRACTION) < 0.01

    train_illicit_rate = data.y[data.train_mask].eq(1).float().mean().item()
    val_illicit_rate = data.y[data.val_mask].eq(1).float().mean().item()
    assert abs(train_illicit_rate - val_illicit_rate) < 0.02


def test_class_weights(loaded):
    data, _ = loaded
    weights = compute_class_weights(data.y, data.train_mask)
    assert weights[1] > weights[0]
