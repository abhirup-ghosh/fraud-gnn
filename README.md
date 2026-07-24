# Fraud-GNN — Graph Neural Network Fraud Detection on the Elliptic Bitcoin Dataset

A two-phase case study: use a Graph Neural Network to flag illicit Bitcoin transactions, then serve
it as a real-time, containerised endpoint with full operational monitoring.

---

## Background & purpose (start here)

**The problem, in plain terms.** Bitcoin is pseudonymous: you can see every payment, but not who is
behind it. Criminals exploit this to move money — ransomware payouts, dark-market sales, scams — by
routing funds through chains of transactions to blend in with ordinary activity. Investigators and
exchanges want to answer one question fast: *does this transaction look illicit?*

**Why a graph, and why a "GNN".** Money laundering is fundamentally about *connections* — who paid
whom, and what the neighbours look like. If we lay the data out as a **graph** (each transaction is a
dot, each payment is a line between two dots), suspicious activity often shows up as suspicious
*neighbourhoods*, not just suspicious individual transactions. A **Graph Neural Network (GNN)** is a
model built for exactly this: instead of judging each transaction in isolation, it lets every
transaction "listen to" its neighbours before deciding — a bit like judging a person partly by the
company they keep. This is the intuition; the technical detail is in the notebook and plan.

**The dataset.** We use the public [**Elliptic** dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set):
~204,000 real Bitcoin transactions, ~234,000 payment links between them, and a set of anonymised
numerical "features" describing each transaction. Some transactions are labelled **licit** (legal) or
**illicit** (criminal); most are **unlabelled**. The features are deliberately anonymised — we don't
know exactly what each number means — so the model has to learn from patterns, not from human-readable
fields.

**Three honest challenges** this project is built around (all confirmed in our analysis):
1. **Needles in a haystack.** Only ~2% of transactions are illicit, and only ~23% are labelled at all.
   A model that blindly says "everything is legal" would be 98% "accurate" and completely useless — so
   we measure success with fraud-appropriate metrics, not raw accuracy.
2. **The past doesn't perfectly predict the future.** The data spans ~1 year in 49 time windows.
   Crucially, a model trained on earlier windows **stops working well** on later ones — for example
   when a real dark market was shut down, the patterns shifted overnight. This is called *concept
   drift*, and it is the single most important operational risk.
3. **It has to run for real.** A model in a notebook helps no one. The end goal is a live service you
   can send a transaction to and get an instant risk score back — with dashboards that tell you when
   the model is degrading, so it can be caught and retrained.

**What this project delivers.** (1) an analysis that pins down the strongest fraud signals, (2) a GNN
that scores transactions, and (3) a locally-runnable, containerised service that serves scores in
real time **and monitors its own health and accuracy** — because, as challenge #2 shows, a fraud model
that isn't watched will quietly fail.

**Who this is for.** Anyone comfortable with the ideas of cryptocurrency, fraud, and machine learning —
you do **not** need deep GNN expertise to follow the README and the narrative in the notebook. Deeper
technical detail lives in [`docs/plan.md`](docs/plan.md) and the code.

---

## Project at a glance

- **Dataset:** [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) —
  203,769 transactions (nodes), 234,355 payment edges, 49 time steps, 166 anonymised features.
- **Environment:** Apple Silicon (M3), Python 3.12, [uv](https://github.com/astral-sh/uv) for deps.
- **Docs:** [`docs/plan.md`](docs/plan.md) (build spec **and** live status/results) ·
  [`docs/deviations.md`](docs/deviations.md) (changes from plan) ·
  [`docs/followup.md`](docs/followup.md) (future work / out of scope).
- **Status:** Phase 1 (EDA) ✅ and Phase 2 (implementation) ✅ both complete. The GNN is built, tested,
  containerised, and monitored — but does **not** beat the Phase-1 RF baseline on test data (see
  Phase 2 below for the honest result and diagnosis).

## Phase 1 — Exploratory Data Analysis ✅ (this phase)

- **[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)** — full EDA with visualisations.
- **[`docs/plan.md`](docs/plan.md)** — the unambiguous, step-by-step Phase-2 build specification.
- `reports/` — exported figures and `eda_summary.json` headline numbers.

### Headline findings

| Signal | Finding | Implication |
|---|---|---|
| Imbalance | 2.2% illicit; only 23% of nodes labelled | class-weighted loss; PR-AUC/F1 metrics; transductive GNN |
| Temporal locality | **100% of edges stay within one time step** (49 isolated slices) | temporal split (train ≤34 / test ≥35); intra-step message passing |
| Topology | sparse DAG; illicit nodes are **lower-degree** (2.0 vs 3.1) | structural signal for the GNN |
| Feature signal | `feat_53/55/89/90` separate classes best (\|d\|≈0.7–1.2) | local features dominate |
| **Concept drift** | test performance **collapses after time step ~43** (dark-market shutdown) | **drift + performance monitoring are mandatory** |

A Random-Forest baseline on the temporal split reaches **ROC-AUC 0.94 / PR-AUC 0.80 / F1 0.80** —
the bar the Phase-2 GNN must beat.

## Phase 2 — Implementation ✅ (see [`docs/plan.md`](docs/plan.md))

Real-time architecture: a GraphSAGE model (PyTorch/PyTorch Geometric) served behind a FastAPI
`/score` endpoint, backed by a Redis feature/adjacency store, all containerised with
`docker-compose`, plus Prometheus + Grafana monitoring across three layers (system health, model
performance, concept drift). Every stage's plan, acceptance criteria, and **actual measured
results** are recorded live in [`docs/plan.md`](docs/plan.md) — it doubles as the project's
build log, not just a prospective spec.

### Model result — an honest negative finding

The GraphSAGE model does **not** beat the Phase-1 Random-Forest baseline on the held-out (temporal)
test set:

| Metric | RF baseline (Phase 1) | GraphSAGE (shipped) |
|---|---|---|
| ROC-AUC | 0.939 | 0.863 |
| PR-AUC | 0.798 | **0.630** |
| F1 | 0.804 | **0.527** |

We diagnosed *why*, rather than just accepting it: the validation split used for early stopping is a
random slice of **non-drifted** training-period data (deliberately, to avoid the drift zone at
design time) — so the checkpoint that gets selected is the one best at predicting normal conditions,
which turns out to be the *opposite* of what generalises to the drift-affected test period. Every
hyperparameter combination tried reaches validation PR-AUC ≈0.98 while test PR-AUC stays in the
0.29–0.63 range — a signature consistent with this explanation, not plain overfitting fixable by
dropout or learning rate. Five of the twelve allowed tuning combinations were tried without closing
the gap. The decision — made explicitly, not silently — was to ship this as a **documented
limitation** rather than open-ended architecture or validation-design work. Full diagnosis, every
combination tried, and two concrete candidate fixes (a temporal validation split; a residual
connection so the model can't dilute the strong local features found in Phase 1) are in
[`docs/deviations.md`](docs/deviations.md) and [`docs/followup.md`](docs/followup.md).

This is a deliberately honest result, reported the way it happened: a tabular baseline beating a GNN
under distribution shift is a real and useful finding in its own right, not a failure to hide.

### The live service

`docker-compose` brings up four containers:
- **api** — FastAPI, serving `/score` (real-time fraud probability), `/health`, `/metrics`, `/feedback`
- **redis** — online feature + adjacency store (per-transaction features and 1-hop neighbours,
  pipelined for low latency)
- **prometheus** — scrapes system-health, model-performance, and concept-drift metrics every 5s
- **grafana** — a fully provisioned dashboard (no manual setup): *System Health*, *Prediction Volume
  & Score Distribution*, *Concept Drift*, *Model Performance*

**Verified end-to-end** on a full clean rehearsal (`make setup && make train && make eval && make up
&& make seed && make loadgen` — not just described, actually run; see `docs/plan.md` §S10/§S11):
- **5,000 real `/score` requests: 0 errors, p95 latency = 11.5ms** (target: <50ms).
- The load generator's drift-inducing phase (feeding real post-dark-market-shutdown transactions)
  pushed the live drift score to **PSI = 0.86** — well past the "significant" alert threshold (0.25),
  reproducing the Phase-1 concept-drift finding live, on the running dashboard.

### Quickstart (Phase 2)

```bash
make setup                    # uv sync
make train && make eval       # train the GNN, print final test metrics
make up                       # docker compose up --build -d (4 containers)
make seed                     # load the Elliptic graph into Redis
make loadgen                  # send 5,000 test requests, verify p95 latency
# Grafana:    http://localhost:3000  (dashboards provisioned automatically)
# Prometheus: http://localhost:9090
# API:        http://localhost:8000/docs
make down                     # tear everything down
```

## Quickstart (Phase 1 — EDA)

```bash
uv sync                                   # install environment
uv run jupyter lab notebooks/01_eda.ipynb # open the EDA
```
