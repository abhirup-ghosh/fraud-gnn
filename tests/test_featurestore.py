"""Requires a local Redis populated via `uv run python -m scripts.load_featurestore`."""
import numpy as np
import pytest
import redis

from fraud_gnn import config as cfg
from fraud_gnn.featurestore import get_features, get_neighbors, get_redis_client, get_subgraph

# Highest-degree node in the Elliptic graph (degree 473; see Phase-1 EDA / data.py), used
# so tests don't need to SCAN all ~200k `nbr:*` keys just to find one.
HIGH_DEGREE_TXID = 2984918


@pytest.fixture(scope="module")
def client():
    c = get_redis_client()
    try:
        c.ping()
    except redis.ConnectionError:
        pytest.skip("no local Redis available")
    if c.dbsize() == 0:
        pytest.skip("feature store not populated; run scripts.load_featurestore first")
    return c


def test_dbsize_is_roughly_two_per_node(client):
    assert client.dbsize() == 2 * cfg.N_NODES


def test_get_subgraph_high_degree_node(client):
    x, edge_index, center_idx = get_subgraph(HIGH_DEGREE_TXID, client=client)

    assert x.shape[0] >= 2
    assert x.shape[1] == cfg.N_FEATURES

    edges = set(map(tuple, edge_index.T.tolist()))
    assert all((b, a) in edges for a, b in edges)  # symmetric
    assert any(center_idx in e for e in edges)  # centre appears


def test_get_features_missing_txid_raises(client):
    with pytest.raises(KeyError):
        get_features(999_999_999_999, client=client)


def test_get_neighbors_returns_list(client):
    neighbors = get_neighbors(HIGH_DEGREE_TXID, client=client)
    assert isinstance(neighbors, list)
    assert len(neighbors) > 0
