"""Redis-backed online feature/adjacency store for real-time neighbourhood lookups."""
import numpy as np
import redis
import torch
from torch_geometric.utils import to_undirected

from fraud_gnn import config as cfg


def get_redis_client(host: str = None, port: int = None) -> redis.Redis:
    return redis.Redis(
        host=host or cfg.REDIS_HOST, port=port or cfg.REDIS_PORT, decode_responses=True
    )


def get_features(tx_id: int, client: redis.Redis = None) -> np.ndarray:
    client = client or get_redis_client()
    key = f"feat:{tx_id}"
    raw = client.hgetall(key)
    if not raw:
        raise KeyError(f"txId {tx_id} not found in feature store")
    values = [float(raw[f"feat_{i}"]) for i in range(1, cfg.N_FEATURES + 1)]
    return np.array(values, dtype=np.float32)


def get_neighbors(tx_id: int, client: redis.Redis = None) -> list[int]:
    client = client or get_redis_client()
    key = f"nbr:{tx_id}"
    members = client.smembers(key)
    if members is None:
        return []
    return [int(m) for m in members]


def get_subgraph(
    tx_id: int, hops: int = 1, client: redis.Redis = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Builds the induced k-hop subgraph around tx_id from the store.

    Returns (x, edge_index, center_idx): x is [n_nodes, N_FEATURES], edge_index is
    [2, n_edges] (undirected, local indices), center_idx is tx_id's row in x.
    """
    client = client or get_redis_client()

    frontier = {tx_id}
    visited = {tx_id}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for nbr in get_neighbors(node, client):
                if nbr not in visited:
                    next_frontier.add(nbr)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    node_ids = [tx_id] + sorted(visited - {tx_id})
    id_to_local = {tx: i for i, tx in enumerate(node_ids)}

    x = np.stack([get_features(tx, client) for tx in node_ids])

    src, dst = [], []
    for tx in node_ids:
        for nbr in get_neighbors(tx, client):
            if nbr in id_to_local:
                src.append(id_to_local[tx])
                dst.append(id_to_local[nbr])

    if src:
        edge_index = to_undirected(torch.tensor([src, dst], dtype=torch.long)).numpy()
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return x, edge_index, id_to_local[tx_id]
