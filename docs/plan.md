# Phase 2 — Implementation Plan (build exactly as written)

This document is a **step-by-step, unambiguous build specification** for a real-time, containerised
Graph Neural Network fraud-detection service on the Elliptic Bitcoin dataset. It is derived from the
Phase-1 EDA (`notebooks/01_eda.ipynb`, findings in `reports/eda_summary.json`).

**Implement the stages in order (§S1 … §S11).** Every file path, dependency, hyperparameter, and
interface is specified. Where a number is given, use that number. Do not add scope that is not listed
here.

## Working conventions (apply to every stage)

1. **Acceptance criteria are binding.** Each stage ends with an **✅ Acceptance criteria** box. Do
   not move to the next stage until every checkbox passes. They are objective and testable.
2. **Commit & push at the end of every stage.** After the acceptance criteria pass, run the exact
   git commands in that stage's **📦 Commit** box. One stage = one commit. Never leave a stage's work
   uncommitted before starting the next.
3. **Log deviations.** If you must build anything differently from this plan (different library,
   changed hyperparameter, altered interface), add a row to [`docs/deviations.md`](deviations.md)
   *in the same commit*. Do not deviate silently.
4. **Defer, don't detour.** If you spot an improvement, add it to *Future improvements* in
   [`docs/followup.md`](followup.md) and keep building the current stage as written. If something is
   explicitly not wanted, add it to *Out of scope* there.
5. All commands are run from the repository root with the `uv` environment active.
6. **Commit & push at the end of every session**, not just every stage. If a session ends mid-stage
   (interrupted, paused, or stopped by the user) with work that doesn't yet meet that stage's
   acceptance criteria, commit and push it anyway — never leave uncommitted work sitting locally
   between sessions. Use a WIP-style message (e.g. `Phase2 S5 WIP: <what's done so far>`) instead of
   the stage's normal commit message, and finish with the stage's real commit once the acceptance
   criteria pass.
7. **Stop after every stage.** Once a stage's acceptance criteria pass and its commit is pushed, do
   **not** automatically start the next stage. Summarise what was done in that stage (what was built,
   what passed, any deviations logged) and then stop and wait. Only begin the next stage once the
   human explicitly asks to continue.

---

## 0. Ground truth from Phase-1 EDA (do not re-derive)

| Fact | Value | Why it matters to the build |
|---|---|---|
| Nodes / edges | 203,769 / 234,355 | Fits in memory on an M3; use **full-graph** training |
| Labelled | 22.85% (4,545 illicit, 42,019 licit) | Rest are `unknown` → **transductive** GNN, masked loss |
| Class balance | 2.23% illicit overall; 9.76% of labelled | **Class-weighted loss**; metrics = **PR-AUC, F1, recall@illicit** |
| Edges cross time steps | **0%** (all intra-step; 49 components) | Message passing is intra-step; use a **temporal split** |
| Temporal split | train `time_step ≤ 34`, test `time_step ≥ 35` | Canonical, leakage-free (val is random-in-train, see §S1) |
| RF baseline (temporal test) | ROC-AUC 0.939, PR-AUC 0.798, **F1 0.804** | **The GNN must beat F1 0.804 / PR-AUC 0.798** |
| Concept drift | perf collapses after `time_step ≈ 43` | **Drift + performance monitoring are mandatory** |
| Top features | `feat_53, feat_55, feat_89, feat_90, feat_47, feat_43` | Sanity-check GNN importances against these |

Feature layout of `elliptic_txs_features.csv` (no header): column 0 = `txId`, column 1 = `time_step`
(1–49), columns 2–166 = 165 anonymised features (`feat_1..feat_165`; `feat_1..feat_93` local,
`feat_94..feat_165` aggregated). All features are already standardised.

---

## 1. Target architecture

```
                         docker-compose (all local, Apple-Silicon friendly)
 ┌──────────────┐   HTTP    ┌────────────────────────┐   lookup   ┌───────────┐
 │  client /    │ ────────▶ │  api  (FastAPI+Uvicorn)│ ─────────▶ │  redis    │
 │  load-gen    │           │  - /score  /health     │            │ feature + │
 └──────────────┘           │  - GraphSAGE (PyTorch) │            │ adjacency │
                            │  - Prometheus /metrics  │            │  store    │
                            └───────────┬────────────┘            └───────────┘
                                        │ scrape
                                ┌───────▼────────┐   dashboards   ┌───────────┐
                                │   prometheus   │ ─────────────▶ │  grafana  │
                                └────────────────┘                └───────────┘
```

Four containers: **api**, **redis**, **prometheus**, **grafana**. Training runs on the host (uses
the M3 **MPS** GPU); artefacts are baked into the api image.

---

## 2. Repository layout to create

```
fraud-gnn/
├─ data/                         # already present (raw Elliptic CSVs) — DO NOT commit
├─ docs/                         # plan.md, deviations.md, followup.md
├─ notebooks/01_eda.ipynb        # already present (Phase 1)
├─ pyproject.toml                # uv-managed (already present; add deps in §S0)
├─ Makefile                      # convenience targets (§S10)
├─ src/fraud_gnn/
│  ├─ __init__.py
│  ├─ config.py                  # all constants/hyperparameters (single source of truth)
│  ├─ data.py                    # load CSVs -> PyG Data + masks
│  ├─ model.py                   # GraphSAGE definition
│  ├─ train.py                   # training loop, early stopping, saves artefacts
│  ├─ evaluate.py                # metrics on temporal test + per-step drift table
│  ├─ baseline.py                # RandomForest baseline (reproduce EDA numbers)
│  ├─ featurestore.py            # build & query Redis feature/adjacency store
│  ├─ drift.py                   # PSI / KS drift score vs training reference
│  └─ serve/
│     ├─ app.py                  # FastAPI app: /score, /health, /metrics
│     ├─ schemas.py              # pydantic request/response models
│     └─ inference.py            # load model + neighbourhood forward pass
├─ artifacts/                    # produced by training (model + reference stats)
├─ tests/                        # test_data, test_model, test_drift, test_api
├─ docker/                       # Dockerfile.api, prometheus.yml, grafana/ provisioning
├─ scripts/                      # load_featurestore.py, loadgen.py
└─ docker-compose.yml
```

---

# Build stages

Each stage is self-contained: **do the work → pass Acceptance criteria → Commit & push → next stage.**

---

## §S0 — Environment & dependencies

**Do:**
```bash
# core DL (Apple Silicon: torch wheels include MPS; PyG core needs no compiled extensions)
uv add "torch>=2.3" "torch-geometric>=2.5"
# serving + store + monitoring + drift
uv add fastapi "uvicorn[standard]" pydantic redis prometheus-client \
       prometheus-fastapi-instrumentator scikit-learn pandas numpy pyarrow
# dev/test
uv add --dev pytest httpx
```
Do **not** install `torch-scatter`/`torch-sparse` (compiled extensions). Use PyG layers that work
without them (`SAGEConv`, `GCNConv` operate on `edge_index` directly). Create empty package files
`src/fraud_gnn/__init__.py` and `src/fraud_gnn/serve/__init__.py`. Create
`src/fraud_gnn/config.py` holding every constant used later: paths, `SEED=42`, split boundaries
(`TRAIN_MAX_STEP=34`, `TEST_MIN_STEP=35`), model hyperparameters (§S2), drift constants (§S6).

**✅ Acceptance criteria**
- [ ] `uv run python -c "import torch, torch_geometric, fastapi, redis, prometheus_client; print('ok')"` prints `ok`.
- [ ] `uv run python -c "import torch; print(torch.backends.mps.is_available())"` runs without error (prints `True` on M3).
- [ ] `uv run python -c "import fraud_gnn.config as c; print(c.SEED)"` prints `42`.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S0: add deps and config scaffolding" && git push
```

---

## §S1 — Data module (`src/fraud_gnn/data.py`)

**Do:** implement `load_elliptic(data_dir) -> (Data, txid_to_idx)` exactly:
1. Read the three CSVs (features `header=None`; assign names per §0).
2. Build `txid_to_idx = {txId: i}` in feature-row order; save mapping.
3. `x = float32[203769,165]` from `feat_1..feat_165`.
4. `y`: `illicit→1, licit→0, unknown→-1` (`long[203769]`).
5. `edge_index`: map both endpoints via `txid_to_idx`; then `edge_index = to_undirected(edge_index)`.
6. `time_step: long[203769]`.
7. Masks (bool `[203769]`): `labelled = y!=-1`; `train_mask = labelled & (time_step<=34)`;
   `test_mask = labelled & (time_step>=35)`; `val_mask` = a **stratified random 15%** carved out of
   `train_mask` with **seed 42** (removed from `train_mask`). (A temporal val slice would sit in the
   drift zone and mislead early stopping.)
8. Add `compute_class_weights(y, mask) -> tensor([w_licit, w_illicit])` with `w_c = n_total/(2*n_c)`.

**✅ Acceptance criteria** (encode these as `tests/test_data.py`, run with `uv run pytest tests/test_data.py -q`)
- [ ] `x.shape == (203769, 165)`, `y.shape == (203769,)`, `time_step.shape == (203769,)`.
- [ ] `edge_index` is symmetric (undirected): every `(a,b)` has a `(b,a)`.
- [ ] `train_mask`, `val_mask`, `test_mask` are pairwise disjoint and contain **no** `unknown` node.
- [ ] `train_mask.sum() + val_mask.sum()` equals the count of labelled nodes with `time_step<=34`.
- [ ] `val` is ~15% (±1%) of the labelled-train nodes and stratified (illicit share within ±2% of train).
- [ ] `compute_class_weights` returns `w_illicit > w_licit`.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S1: data loader, masks, class weights + tests" && git push
```

---

## §S2 — Model (`src/fraud_gnn/model.py`)

**Do:** implement a 2-layer **GraphSAGE** (inductive → can serve unseen transactions):
```
x -> SAGEConv(165,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.4)
  -> SAGEConv(128,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.4)
  -> Linear(128,2)   (logits)
```
Hyperparameters in `config.py`: `HIDDEN=128, DROPOUT=0.4, LR=5e-3, WEIGHT_DECAY=5e-4, EPOCHS=300,
PATIENCE=30, SEED=42, AGGR="mean"`.

**✅ Acceptance criteria** (`tests/test_model.py`)
- [ ] `FraudSAGE(165,128,2).forward(x, edge_index)` on a random 50-node graph returns shape `[50,2]`.
- [ ] Output requires grad and `loss.backward()` populates parameter gradients (differentiable).
- [ ] Parameter count is deterministic across two constructions with `SEED=42`.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S2: GraphSAGE model + tests" && git push
```

---

## §S3 — Baseline (`src/fraud_gnn/baseline.py`)

**Do:** reproduce the EDA Random-Forest on the temporal split so the GNN has a bar to beat. Fit
`RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)` on
labelled train nodes, evaluate on labelled test nodes; print ROC-AUC, PR-AUC, F1, precision, recall.

**✅ Acceptance criteria**
- [ ] `uv run python -m fraud_gnn.baseline` prints **F1 in [0.79, 0.82]** and **PR-AUC in [0.78, 0.82]** (matches EDA: F1≈0.804, PR-AUC≈0.798).

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S3: RandomForest baseline reproducing EDA numbers" && git push
```

---

## §S4 — Training (`src/fraud_gnn/train.py`)

**Do:**
1. Seed `torch/numpy/random` with 42; device = `mps` if available else `cpu`.
2. `load_elliptic`; move to device; **full-batch** training (whole graph each step).
3. Loss = `CrossEntropyLoss(weight=class_weights)` on `train_mask` only.
   Optimiser = `Adam(lr=LR, weight_decay=WEIGHT_DECAY)`.
4. Each epoch: forward → masked loss → step; compute **val PR-AUC** on `val_mask`. Early-stop on best
   val PR-AUC with `PATIENCE=30`; keep best weights.
5. Save artefacts: `artifacts/model.pt` (state_dict), `artifacts/metadata.json`
   (`in_dim, hidden, dropout, aggr, best_val_pr_auc, best_epoch, f1_optimal_threshold,
   feature_columns, model_version`), `artifacts/txid_to_idx.json`, and
   `artifacts/reference_stats.parquet` (per-feature mean/std + 10 quantiles over **training** nodes).
6. `model_version` = short timestamp or git commit hash.

**✅ Acceptance criteria**
- [ ] `uv run python -m fraud_gnn.train` completes and writes all four artefacts under `artifacts/`.
- [ ] Training uses MPS on M3 (log line prints the device; must be `mps` when available).
- [ ] `metadata.json` contains a numeric `best_val_pr_auc` and an `f1_optimal_threshold` in (0,1).
- [ ] Run is reproducible: two runs give `best_val_pr_auc` within ±0.02.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S4: full-batch GraphSAGE training + artefact export" && git push
```
> Note: `artifacts/` is git-ignored; commit the training code, not the weights.

---

## §S5 — Evaluation & the acceptance gate (`src/fraud_gnn/evaluate.py`)

**Do:** `evaluate(model, data)` prints/returns:
- Test-mask metrics: ROC-AUC, PR-AUC, F1, precision, recall for the illicit class at threshold 0.5,
  **and** at the F1-optimal threshold from `metadata.json`.
- Per-time-step table for steps 35–49: `time_step, n, n_illicit, roc_auc, f1` (the post-step-43
  collapse must be visible).

**✅ Acceptance criteria (the model gate)**
- [ ] `uv run python -m fraud_gnn.evaluate` reports **test PR-AUC ≥ 0.80 AND F1 ≥ 0.80** (≥ RF baseline).
- [ ] The per-step table is printed for all of steps 35–49 and shows degradation after step ~43.
- [ ] If the gate is not met, tune **only** `HIDDEN∈{128,256}`, `DROPOUT∈{0.3,0.4,0.5}`,
      `LR∈{1e-3,5e-3}` (no architectural change) and log the chosen values in `docs/deviations.md`.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S5: evaluation, per-step drift table, acceptance gate met" && git push
```

---

## §S6 — Concept-drift monitor (`src/fraud_gnn/drift.py`)

**Do:** implement a PSI-based drift monitor (directly targets the Phase-1 drift finding).
- Load `artifacts/reference_stats.parquet`. Keep a sliding buffer (size `DRIFT_BUFFER=1000`) of
  incoming feature vectors.
- Every `DRIFT_WINDOW=200` observations, compute **PSI** per feature (buffer vs reference quantiles);
  `drift_score = mean(PSI over the 6 top features from §0)`.
- Thresholds: PSI `<0.1` stable, `0.1–0.25` moderate (warn), `>0.25` significant (alert).

**✅ Acceptance criteria** (`tests/test_drift.py`)
- [ ] PSI ≈ 0 (`< 1e-6`) when the buffer distribution equals the reference.
- [ ] PSI `> 0.25` when the buffer is shifted by `+3σ` on the monitored features.
- [ ] `drift_score` is a finite float for a mixed buffer; monitor never raises on unseen feature scale.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S6: PSI concept-drift monitor + tests" && git push
```

---

## §S7 — Feature store (`src/fraud_gnn/featurestore.py`, `scripts/load_featurestore.py`)

**Do:** Redis online store for real-time neighbourhood lookups.
- `scripts/load_featurestore.py`: per `txId` write hash `feat:{txId}` (165 values + `time_step`) and
  set `nbr:{txId}` (undirected neighbour txIds). Use a pipeline, batch 5,000. Host from env
  `REDIS_HOST` (default `localhost`).
- `featurestore.py`: `get_features(txId)->np.ndarray[165]`, `get_neighbors(txId)->list[int]`,
  `get_subgraph(txId, hops=1)->(x, edge_index, center_idx)`. Missing txId raises `KeyError`.

**✅ Acceptance criteria**
- [ ] With a local Redis running, `uv run python -m scripts.load_featurestore` loads all 203,769 nodes; `DBSIZE` ≈ 2× node count (feat + nbr keys).
- [ ] `get_subgraph(txId)` for a known high-degree node returns `x` with ≥2 rows and a symmetric `edge_index` including the centre.
- [ ] `get_features` on an absent txId raises `KeyError`.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S7: Redis feature/adjacency store + loader" && git push
```

---

## §S8 — Inference & API (`src/fraud_gnn/serve/`)

**Do:**
- `inference.py`: load `FraudSAGE` from `artifacts/`, `model.eval()`. `score_known(txId)` (pull
  1-hop subgraph from store → forward → `P(illicit)` of centre) and
  `score_inductive(features[165], neighbor_features)` (star subgraph → forward).
- `schemas.py` (pydantic):
  `ScoreRequest = {txId?:int, features?:list[float]/len165, neighbor_features?:list[list[float]]}`;
  `ScoreResponse = {txId, illicit_probability, label, threshold, model_version, latency_ms, drift_score}`.
  Routing: `txId`→known; elif `features`→inductive; else HTTP 422.
- `app.py` (FastAPI): `GET /health` → `{status,model_version,uptime_s}`; `POST /score` → `ScoreResponse`
  (also updates drift monitor + increments metrics); `GET /metrics` → Prometheus (instrument with
  `prometheus_fastapi_instrumentator` **plus** custom metrics in §S9). Threshold from `metadata.json`
  (fallback 0.5). Startup: connect Redis (`REDIS_HOST`, default `redis`), load model + reference stats.

**✅ Acceptance criteria** (`tests/test_api.py`, monkeypatched model + fake store, `httpx`)
- [ ] `GET /health` → 200 with a `model_version`.
- [ ] `POST /score` with `{txId}` and with `{features:[…165…]}` both return a valid `ScoreResponse` (prob in [0,1]).
- [ ] `POST /score` with neither field → HTTP 422.
- [ ] `GET /metrics` exposes `fraud_drift_psi` and the default HTTP latency metrics.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S8: FastAPI /score /health /metrics + inference paths + tests" && git push
```

---

## §S9 — Monitoring metrics (three layers)

**Do:** wire the metrics that the dashboards consume.
- **A. System health** (free via instrumentator): request rate, p50/p95/p99 latency, error rate,
  in-flight requests, process CPU/memory.
- **B. Model performance** (custom): `fraud_predictions_total{label=}` (Counter),
  `fraud_score_histogram` (Histogram of `P(illicit)`), `fraud_flagged_ratio` (Gauge, rolling share
  predicted illicit over last 1000). Optional `POST /feedback {txId, true_label}` maintaining rolling
  `fraud_rolling_precision/recall/f1` (Gauges).
- **C. Concept drift**: expose `fraud_drift_psi` (Gauge) from §S6; include `drift_score` in every
  `ScoreResponse`.

**✅ Acceptance criteria**
- [ ] `GET /metrics` after N>200 `/score` calls exposes non-zero `fraud_predictions_total`, a populated `fraud_score_histogram`, and a `fraud_drift_psi` value.
- [ ] `POST /feedback` updates the rolling precision/recall/f1 gauges (if implemented).

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S9: system/model/drift Prometheus metrics" && git push
```

---

## §S10 — Containerisation, Makefile & load generator

**Do:**
- `docker/Dockerfile.api`: multi-stage `python:3.12-slim`; builder does `uv sync --no-dev`; runtime
  copies venv + `src/` + `artifacts/`; `CMD uvicorn fraud_gnn.serve.app:app --host 0.0.0.0 --port 8000`.
  Use **CPU torch** in the image (inference on a small subgraph is sub-ms on CPU).
- `docker-compose.yml` services: `redis` (redis:7-alpine, healthcheck `redis-cli ping`),
  `api` (build Dockerfile.api, port 8000, `depends_on: redis` healthy, `REDIS_HOST=redis`),
  `prometheus` (prom/prometheus, mount `docker/prometheus.yml`, port 9090, scrape `api:8000/metrics`
  every 5s), `grafana` (grafana/grafana, port 3000, mount provisioning; anonymous admin for local demo).
- `docker/grafana/` provisioning: one `prometheus` datasource + one dashboard JSON with rows
  *System Health*, *Prediction Volume & Score Distribution*, *Concept Drift* (panel plots
  `fraud_drift_psi` with 0.1/0.25 threshold lines), *Model Performance (feedback)*.
- `scripts/loadgen.py --n 5000`: send a mix of known-`txId` and inductive requests to `/score`,
  including a drift-inducing phase (feed later-time-step nodes) so the drift panel visibly moves.
- `Makefile` targets: `setup, features, train, eval, baseline, test, up, seed, loadgen, down`
  (exact commands as in prior spec).

**✅ Acceptance criteria** (§ Definition of done for the system)
- [ ] `make up` brings up **4 healthy containers**; `GET http://localhost:8000/health` → 200.
- [ ] After `make seed`, `POST /score {txId}` returns `illicit_probability`, `drift_score`, `model_version` with **p95 latency < 50 ms** (verify via `make loadgen`).
- [ ] Grafana at `http://localhost:3000` shows live *System Health*, *Score Distribution*, and *Concept Drift* panels updating during `make loadgen`.
- [ ] `make down` tears everything down cleanly (`-v`).

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S10: docker-compose stack, Grafana dashboards, loadgen, Makefile" && git push
```

---

## §S11 — End-to-end verification & wrap-up

**Do:** run the whole pipeline start-to-finish on a clean checkout: `make setup && make train &&
make eval && make up && make seed && make loadgen`. Update `README.md` Phase-2 section with final
test metrics and screenshots/links. Fold every row of `docs/deviations.md` back into this `plan.md`
so plan and delivered system agree; review `docs/followup.md`.

**✅ Acceptance criteria (project done)**
- [ ] `make test` is green (all of test_data/model/drift/api pass).
- [ ] Full Definition-of-Done from §S5 (gate) and §S10 (system) all pass on a clean run.
- [ ] `README.md` reports the final GNN test PR-AUC/F1 and confirms it beats the RF baseline.
- [ ] `docs/deviations.md` is reconciled into `plan.md`; `docs/followup.md` reviewed.

**📦 Commit**
```bash
git add -A && git commit -m "Phase2 S11: end-to-end verified; docs reconciled" && git push
```

---

## Build order (do not reorder)

`§S0 → §S1 → §S2 → §S3 → §S4 → §S5 → §S6 → §S7 → §S8 → §S9 → §S10 → §S11`

Each arrow crosses a passed acceptance-criteria box **and** a pushed commit.
