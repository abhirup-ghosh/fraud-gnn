"""Loads the Elliptic graph into Redis as per-node feature hashes + adjacency sets."""
from collections import defaultdict

from fraud_gnn import config as cfg
from fraud_gnn.data import load_elliptic
from fraud_gnn.featurestore import get_redis_client

BATCH_SIZE = 5000


def build_adjacency(edge_index, n_nodes: int) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adjacency[s].append(d)
    return adjacency


def load_featurestore():
    data, txid_to_idx = load_elliptic()
    idx_to_txid = {i: tx for tx, i in txid_to_idx.items()}
    n_nodes = data.x.shape[0]

    adjacency = build_adjacency(data.edge_index, n_nodes)

    client = get_redis_client()
    client.flushdb()

    x = data.x.numpy()
    time_step = data.time_step.numpy()

    pipe = client.pipeline(transaction=False)
    pending = 0
    for i in range(n_nodes):
        tx_id = idx_to_txid[i]
        feat_mapping = {f"feat_{j}": float(x[i, j - 1]) for j in range(1, cfg.N_FEATURES + 1)}
        feat_mapping["time_step"] = int(time_step[i])
        pipe.hset(f"feat:{tx_id}", mapping=feat_mapping)

        neighbor_txids = [idx_to_txid[n] for n in adjacency.get(i, [])]
        if neighbor_txids:
            pipe.sadd(f"nbr:{tx_id}", *neighbor_txids)

        pending += 1
        if pending >= BATCH_SIZE:
            pipe.execute()
            pipe = client.pipeline(transaction=False)
            pending = 0

    if pending:
        pipe.execute()

    print(f"loaded {n_nodes} nodes into feature store (DBSIZE={client.dbsize()})")


if __name__ == "__main__":
    load_featurestore()
