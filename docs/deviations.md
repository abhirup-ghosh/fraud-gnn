# Deviations Log

This file records **every deviation from [`plan.md`](plan.md)** made during Phase-2 implementation:
where the build differed from the written spec, and why. At the end of the project we will fold these
deviations back into `plan.md` so the plan and the delivered system agree.

**How to use:** whenever you implement something differently from what `plan.md` says (different
library, changed hyperparameter, altered interface, extra/removed step), add a dated row below. Do
not silently deviate — log it.

| Date | Plan section | What the plan said | What was actually done | Reason |
|------|--------------|--------------------|------------------------|--------|
| 2026-07-24 | §S4 (`train.py`) | "Training uses MPS on M3 (log line prints the device; must be `mps` when available)" | Training runs on **CPU** by default. MPS is still detected and logged (`mps available: True/False`), but `get_device()` returns `cpu` unless `FRAUD_GNN_FORCE_MPS_TRAIN=1` is set. | Empirically, MPS gives **non-deterministic, unstable** full-batch training on this graph: three runs with identical seed/config produced val PR-AUC of 0.6685, 0.9795, and (with `torch.use_deterministic_algorithms(warn_only=True)`) 0.8284 / 0.8573 — one run's loss diverged outright after epoch ~10. This violates §S4's own reproducibility criterion ("two runs give `best_val_pr_auc` within ±0.02"), which conflicts with the MPS requirement on this hardware/PyTorch/PyG combination (root cause: `SAGEConv`'s scatter-reduce aggregation is not deterministic on the MPS backend). CPU training is bit-for-bit reproducible (two runs: `best_val_pr_auc=0.9828`, `best_epoch=155` identical to 4 decimal places) and completes in ~2 minutes for this dataset size (203k nodes), so the reproducibility guarantee was prioritised over MPS. |
| 2026-07-24 | n/a (tooling, not a plan section) | — | Added `conftest.py` at repo root (inserts `src/` onto `sys.path`) so `pytest` reliably finds the `fraud_gnn` package. | The local `.venv` (created by `uv`) contains a `_virtualenv.pth` site patch that, on this machine, non-deterministically drops the editable install's `src/` path from `sys.path` depending on `.pth` processing order (`_editable_impl_fraud_gnn.pth` sorts before `_virtualenv.pth` alphabetically and gets filtered out). Symptom: `import fraud_gnn` intermittently raised `ModuleNotFoundError` even with no code changes between runs. `conftest.py` sidesteps the issue for tests; module invocations (`uv run python -m fraud_gnn.x`) remain occasionally affected — if hit, `uv sync` or re-running resolves it. Not a code defect; a local environment quirk worth knowing about. |
