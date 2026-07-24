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


def _pipeline_smembers(tx_ids: list[int], client: redis.Redis) -> dict[int, list[int]]:
    """Batch SMEMBERS for many tx_ids in a single round trip."""
    if not tx_ids:
        return {}
    pipe = client.pipeline(transaction=False)
    for tx in tx_ids:
        pipe.smembers(f"nbr:{tx}")
    results = pipe.execute()
    return {tx: [int(m) for m in members] for tx, members in zip(tx_ids, results)}


def _pipeline_hgetall(tx_ids: list[int], client: redis.Redis) -> dict[int, np.ndarray]:
    """Batch HGETALL for many tx_ids in a single round trip."""
    if not tx_ids:
        return {}
    pipe = client.pipeline(transaction=False)
    for tx in tx_ids:
        pipe.hgetall(f"feat:{tx}")
    results = pipe.execute()
    out = {}
    for tx, raw in zip(tx_ids, results):
        if not raw:
            raise KeyError(f"txId {tx} not found in feature store")
        out[tx] = np.array(
            [float(raw[f"feat_{i}"]) for i in range(1, cfg.N_FEATURES + 1)], dtype=np.float32
        )
    return out


def get_subgraph(
    tx_id: int, hops: int = 1, client: redis.Redis = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Builds the induced k-hop subgraph around tx_id from the store.

    Redis calls are batched via pipelines (one round trip per BFS layer, one for all
    feature lookups) rather than one round trip per node — a single-round-trip-per-node
    approach made scoring a high-degree node (up to 473 neighbours) take ~475ms; this
    keeps p95 request latency low regardless of node degree.

    Returns (x, edge_index, center_idx): x is [n_nodes, N_FEATURES], edge_index is
    [2, n_edges] (undirected, local indices), center_idx is tx_id's row in x.
    """
    client = client or get_redis_client()

    if not client.exists(f"feat:{tx_id}"):
        raise KeyError(f"txId {tx_id} not found in feature store")

    frontier = {tx_id}
    visited = {tx_id}
    neighbor_cache: dict[int, list[int]] = {}
    for _ in range(hops):
        nbrs_map = _pipeline_smembers(sorted(frontier), client)
        neighbor_cache.update(nbrs_map)
        next_frontier = set()
        for nbrs in nbrs_map.values():
            for nbr in nbrs:
                if nbr not in visited:
                    next_frontier.add(nbr)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    node_ids = [tx_id] + sorted(visited - {tx_id})
    id_to_local = {tx: i for i, tx in enumerate(node_ids)}

    features_map = _pipeline_hgetall(node_ids, client)
    x = np.stack([features_map[tx] for tx in node_ids])

    missing = [tx for tx in node_ids if tx not in neighbor_cache]
    neighbor_cache.update(_pipeline_smembers(missing, client))

    src, dst = [], []
    for tx in node_ids:
        for nbr in neighbor_cache.get(tx, []):
            if nbr in id_to_local:
                src.append(id_to_local[tx])
                dst.append(id_to_local[nbr])

    if src:
        edge_index = to_undirected(torch.tensor([src, dst], dtype=torch.long)).numpy()
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return x, edge_index, id_to_local[tx_id]
