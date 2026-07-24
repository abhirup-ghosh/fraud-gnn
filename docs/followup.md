# Follow-up

Two running lists maintained throughout the project. Add to them as you go — do **not** act on them
during the current stage.

---

## 1. Future improvements

Ideas we recognise as worth doing but deliberately defer (we note them at the stage where the idea
arises, then keep building). Each entry: what, why it helps, and which `plan.md` stage it relates to.

| Idea | Why it helps | Related plan stage | Added on |
|------|--------------|--------------------|----------|
| Root-cause the `uv`/`_virtualenv.pth` `sys.path` flakiness properly (or recreate `.venv` without the `virtualenv`-style patch) instead of the `conftest.py`/`PYTHONPATH=src` workaround | Removes an occasional `ModuleNotFoundError` gotcha for anyone re-running commands outside the eventual Makefile | §S4 (env tooling) | 2026-07-24 |
| Investigate why `SAGEConv` scatter-reduce is non-deterministic on PyTorch's MPS backend (file an upstream issue / retest on a future PyTorch release) and re-enable MPS training if fixed | Would restore GPU-accelerated training on the M3 without sacrificing reproducibility | §S4 (`train.py`) | 2026-07-24 |
| Try a temporal `val_mask` (e.g. the last few train time steps, 30–34) instead of a random 15% slice of train, so early stopping selects a checkpoint that generalizes to drift rather than one that's merely best on non-drifted data | Directly targets the diagnosed root cause of the §S5 gate failure (val distribution doesn't resemble the drifted test distribution) | §S1 (`data.py`) / §S5 gate | 2026-07-24 |
| Add a residual/skip path from raw input features to the model's output layer (e.g. concatenate `x` with the final `SAGEConv` embedding before the linear head) | Phase-1 EDA found very strong local features (`feat_53/55/89/90`); 2-layer mean-aggregation may be diluting them for low-degree nodes (illicit nodes average degree 2.0) — a residual path would let the model use them directly, undiluted | §S2 (`model.py`) / §S5 gate | 2026-07-24 |
| Finish the remaining 7 of 12 allowed hyperparameter combos for the §S5 gate ((128,0.4,1e-3), (128,0.3,1e-3), (256,0.4,5e-3), (256,0.4,1e-3), (256,0.5,5e-3), (256,0.5,1e-3), (256,0.3,1e-3)) if the two ideas above don't fully close the gap | Completes the originally-planned tuning sweep; low priority since the observed pattern (precision collapse regardless of combo) suggests it's unlikely to help alone | §S5 gate | 2026-07-24 |
| Slim `docker/Dockerfile.api`: split EDA-only deps (jupyter, matplotlib, seaborn, networkx, scikit-learn) into a separate `uv` dependency group excluded from the serving image, and/or point torch at the CPU-only wheel index (the default PyPI linux wheel pulls in `triton`/CUDA-adjacent packages unused on CPU) | Meaningfully smaller, faster-building serving image; current image works correctly but is heavier than necessary | §S10 (`Dockerfile.api`) | 2026-07-24 |

---

## 2. Out of scope

Things we have consciously decided **not** to do in this project, so nobody wastes time on them or
mistakes their absence for an oversight. Each entry: what, and why it's excluded.

| Excluded item | Why it's out of scope |
|---------------|------------------------|
| Cloud / Kubernetes deployment | Target is a **local, single-machine** (Apple M3) demo via docker-compose. |
| Distributed / multi-GPU training | Dataset fits full-batch in memory on one M3; not needed. |
| Real Bitcoin ingestion from a live node | We use the static Elliptic dataset; live-chain ingestion is a separate project. |
| De-anonymising the features | Elliptic features are intentionally anonymised; we treat them as opaque signals. |
| Beating published state-of-the-art benchmarks | Goal is a **complete, monitored, deployable** pipeline, not a leaderboard score. |
