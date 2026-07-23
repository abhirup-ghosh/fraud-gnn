import torch

from fraud_gnn import config as cfg
from fraud_gnn.model import FraudSAGE


def _random_graph(n_nodes=50, n_edges=150, in_dim=165, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, in_dim, generator=g)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    return x, edge_index


def test_forward_shape():
    torch.manual_seed(cfg.SEED)
    model = FraudSAGE(in_dim=165, hidden=128, out_dim=2)
    x, edge_index = _random_graph()
    out = model(x, edge_index)
    assert out.shape == (50, 2)


def test_differentiable():
    torch.manual_seed(cfg.SEED)
    model = FraudSAGE(in_dim=165, hidden=128, out_dim=2)
    x, edge_index = _random_graph()
    out = model(x, edge_index)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_deterministic_param_count():
    torch.manual_seed(cfg.SEED)
    model_a = FraudSAGE(in_dim=165, hidden=128, out_dim=2)
    torch.manual_seed(cfg.SEED)
    model_b = FraudSAGE(in_dim=165, hidden=128, out_dim=2)
    n_a = sum(p.numel() for p in model_a.parameters())
    n_b = sum(p.numel() for p in model_b.parameters())
    assert n_a == n_b
