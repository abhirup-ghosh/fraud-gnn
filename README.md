# Fraud-GNN — Graph Neural Network Fraud Detection on the Elliptic Bitcoin Dataset

## 1. What this project does

**In one sentence:** this project trains a graph-based AI model to spot fraudulent Bitcoin
transactions, then serves it as a live, monitored web service — the whole pipeline from raw data to
a running, observable system, not just a notebook.

Concretely, it takes ~200,000 real (anonymised) Bitcoin transactions, some of which are known to be
tied to criminal activity, builds a model that scores new transactions for how suspicious they look,
and wraps that model in the infrastructure a real deployment would need: a live API, a fast lookup
store, and dashboards that watch whether the model is still working correctly.

**Who this is for:** anyone comfortable with the general ideas of cryptocurrency, fraud, and machine
learning. You do **not** need to already know what a graph neural network is — the explanations below
are built for that. If you want the expert-level detail (exact hyperparameters, every test, every
measured number), it's one level down in [`docs/plan.md`](docs/plan.md); this document stays at the
level of "what was built, what it found, and whether it worked."

---

## 2. The problem, explained simply

Bitcoin is *pseudonymous*: every payment is publicly visible, but nothing tells you who's actually
behind it. Criminals exploit this — ransomware payouts, dark-market sales, laundering — by moving
funds through chains of transactions designed to blend into ordinary activity. Investigators and
exchanges want to answer one question, fast: *does this transaction look illicit?*

**Why connections matter.** Judging a transaction purely on its own numbers (amount, fee, timing)
misses something important: fraud is often visible in the *company a transaction keeps*, not just its
own attributes. If you draw the data out as a **graph** — every transaction is a dot, every payment
between two transactions is a line connecting two dots — suspicious activity often shows up as
suspicious *neighbourhoods*, clusters of connected transactions that behave unusually together, not
just individually unusual transactions in isolation.

**Why a graph neural network, specifically.** A **Graph Neural Network (GNN)** is a model built to
use exactly that structure. Instead of judging each transaction alone, it lets every transaction
"listen to" its immediate neighbours in the graph before making a decision — a bit like judging a
person partly by the company they keep, rather than purely on their own resume. That's the whole
intuition; the rest of this project is about whether that intuition actually pays off in practice (see
§5 for the honest answer).

---

## 3. The data, in plain terms

This project uses the public [**Elliptic** dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set):
real Bitcoin transaction data, released for exactly this kind of research. In plain terms, it's three
things stitched together:

- **~204,000 transactions** — each with a set of anonymised numerical properties. They're
  anonymised deliberately: we don't know what each number literally means, so any model has to learn
  from statistical patterns, not from human-readable fields.
- **~234,000 links** between transactions — who paid whom, forming the graph described in §2.
- **A label for some transactions** — *licit* (legitimate), *illicit* (criminal), or nothing at all.

**Why most transactions are unlabelled, and why it matters.** Only about **1 in 4** transactions has
any label at all — the rest are simply unknown, because nobody has gone through and confirmed them
one way or the other. This is normal for real-world fraud data (nobody labels the entire history of
Bitcoin), but it means the model has to be trained carefully: it can *see* the unlabelled
transactions (their properties and their position in the graph), just not learn from a "correct
answer" for them.

**Three honest challenges this project is built around:**

1. **Fraud is a needle in a haystack, and most of the haystack isn't even labelled.** A model that
   blindly guesses "everything is fine" would look deceptively accurate while being completely
   useless — so success has to be measured with fraud-appropriate yardsticks, not raw accuracy.
2. **The past doesn't reliably predict the future.** A model trained on older data can quietly stop
   working on newer data — sometimes gradually, sometimes overnight, for example when a real
   criminal marketplace gets shut down and behaviour shifts abruptly. This is called *concept drift*,
   and it turns out to be the single most important operational risk in this entire project.
3. **A model sitting in a notebook helps nobody.** The point is a live service: send it a
   transaction, get a risk score back fast enough to be useful, with a way to notice — automatically —
   if the model starts drifting away from reality.

---

## 4. How it works, at a glance

The project has two phases: **analysis first, then a live service** built on what the analysis found.

```
  Phase 1: Analysis                        Phase 2: Live service
  ──────────────────                       ──────────────────────

  raw transaction data                     new transaction arrives
         │                                          │
         ▼                                          ▼
  explore it, find the                      look up its position in
  strongest fraud signals                   the transaction graph
         │                                          │
         ▼                                          ▼
  write an exact build plan          ──▶     ask the GNN model:
  for the live service                       "how suspicious is this,
                                              given what it's linked to?"
                                                     │
                                                     ▼
                                              risk score + decision
                                                     │
                                                     ▼
                                              dashboards: is the
                                              service healthy, accurate,
                                              and still trustworthy?
```

**The smart part vs. the plumbing.** Only one piece of this system is genuinely "smart": the GNN
model itself, which makes the actual judgement call. Everything else — the web server, the fast
lookup store, the metrics, the dashboards — is *plumbing*. Its entire job is to get a real
transaction to the model quickly, and to notice, automatically, the moment the model can no longer be
trusted. Both halves matter: a smart model with no plumbing is a science project; plumbing with no
smart model is pointless.

---

## 5. What we found

### Phase 1 — the analysis

- **Fraud is rare and mostly unlabelled.** About 1 in 45 transactions is confirmed illicit; three
  in four have no label at all.
- **Fewer connections, not more, is the fraud signal.** Counter-intuitively, transactions that turn
  out to be fraudulent tend to have *fewer* connections in the graph than legitimate ones — useful,
  if not what you'd guess going in.
- **A handful of properties do most of the work.** A small number of the anonymised transaction
  properties separate fraud from non-fraud very cleanly, on their own, before any graph reasoning is
  even applied.
- **The past stops predicting the future, sharply.** Every model tried — including the one
  eventually shipped — shows a performance cliff at a specific point in the transaction timeline,
  lining up with a real-world event (a dark-market marketplace being shut down, per public reporting
  on the underlying data). This is the finding the entire monitoring system in §6 exists to catch.

### Phase 2 — the live service, and an honest result

**In one sentence:** a simpler, older-style model (a Random Forest) beat the graph-based model on
held-out data, and that's reported here plainly rather than smoothed over.

| Metric | Random Forest (simpler baseline) | GNN (the model actually shipped) |
|---|---|---|
| ROC-AUC | 0.939 | 0.863 |
| PR-AUC | 0.798 | **0.630** |
| F1 | 0.804 | **0.527** |

**Why, in short:** picking the "best" version of the GNN during training meant choosing the version
that did best on a held-out slice of *normal* data — deliberately excluding the "weird,
post-marketplace-shutdown" data, which sounds sensible. In practice, that backfired: it meant
consistently picking the version of the model that was best at handling normal conditions, which
turned out to be close to the opposite of what actually mattered once evaluated against the genuinely
unusual test period. The full technical diagnosis, every combination of settings tried, and the two
concrete ideas for actually fixing it, are in [`docs/plan.md`](docs/plan.md) (§S5) and
[`docs/followup.md`](docs/followup.md) — deliberately not hidden in this document.

A tabular model beating a graph model under exactly this kind of shift is a real, useful finding in
its own right, not a failure to apologise for.

---

## 6. The live service, explained

Four small programs run together, each with one job:

- **The front door** (`api`) — a web server that receives a transaction, asks the model for a risk
  score, and answers back — this is the only piece anything outside the system talks to directly.
- **The memory** (`redis`) — holds every transaction's properties and its immediate neighbours in
  the graph, so answering "who is this transaction connected to?" is a fast lookup, not a re-scan of
  the whole dataset.
- **The gauges** (`prometheus`) — continuously records how busy the service is, how confident its
  predictions are, and — critically — how much incoming traffic still resembles what the model was
  trained on.
- **The dashboard** (`grafana`) — turns those gauges into screens a human would actually watch, with
  no manual setup required; it's provisioned automatically.

**What "real time" means here.** Send a transaction, get a risk score back in well under a tenth of a
second: in testing, **95 out of 100 requests came back in about 11 milliseconds or less** (for
comparison, a single eye-blink takes roughly 100–150 milliseconds). That was measured by actually
firing 5,000 real requests at the running service, not estimated.

**What the drift monitor is watching for, and why it's necessary.** One of the gauges specifically
tracks *concept drift*: is today's traffic starting to look statistically different from the data the
model was trained on? This isn't a theoretical concern added for completeness — §5 found exactly this
kind of shift already happened once in the historical data, sharply and specifically. A model that
isn't watched for this will quietly keep giving confident answers while being wrong, which is worse
than no model at all. In testing, deliberately feeding the service transactions from the "after the
shift" period pushed this gauge well past its alert threshold — live, on the dashboard — showing the
monitoring actually catches the exact failure mode Phase 1 uncovered.

---

## 7. Try it yourself

```bash
make setup                    # install the Python environment
make train                    # train the GNN (reproducible, ~2 minutes on CPU)
make eval                     # print the model's final test-set numbers

make up                       # start all 4 services in Docker
make seed                     # load the transaction graph into the fast lookup store
make loadgen                  # send 5,000 real test requests, report latency

# then open:
#   http://localhost:8000/docs   — try the API directly (interactive docs)
#   http://localhost:9090        — Prometheus (raw metrics)
#   http://localhost:3000        — Grafana (the dashboard, no login needed)

make down                     # stop everything, clean up
```

**What you should expect to see.** The Grafana dashboard (`http://localhost:3000`) opens straight to
a page titled *Fraud-GNN Monitoring* with four rows, top to bottom: **System Health** (request rate,
latency, CPU/memory — general web-service vital signs), **Prediction Volume & Score Distribution**
(how many transactions are being flagged, and how confident the model is), **Concept Drift** (the
gauge described in §6, with coloured threshold lines at "moderate" and "significant" drift), and
**Model Performance** (accuracy metrics, if feedback on real outcomes is supplied via `/feedback`).
Running `make loadgen` while the dashboard is open makes every panel move in real time — including,
during the load generator's deliberate "drift phase," the Concept Drift panel visibly crossing into
its alert zone.

If you'd rather start with the analysis than the live service, the Phase-1 notebook is simpler:

```bash
uv sync
uv run jupyter lab notebooks/01_eda.ipynb
```

---

## 8. Honest limitations & what's next

- **The shipped model doesn't beat the simpler baseline** (§5) — the headline limitation of this
  project, diagnosed but not yet fixed. Two concrete fixes are identified but deliberately deferred:
  validating the model on data that actually resembles the "shifted" test conditions, and adjusting
  the model architecture so it can't dilute its strongest raw signals. Both are recorded in
  [`docs/followup.md`](docs/followup.md).
- **Not every possible model configuration was tried.** Of the tuning options allowed for fixing the
  gate above, 5 of 12 were tried before deciding to stop and report the honest result rather than
  keep searching (see `docs/plan.md` §S5 for exactly which ones, and why continuing looked unlikely
  to help).
- **The training hardware's GPU (Apple's "MPS" backend) is deliberately unused.** It produced
  wildly inconsistent results between otherwise-identical training runs, so training runs on CPU
  instead — slightly slower, but trustworthy. Worth re-testing on a future PyTorch release
  (`docs/followup.md`).
- **The container image is heavier than it needs to be.** It correctly runs everything needed, but
  currently includes some analysis-only tooling it doesn't use in production — a known, low-priority
  cleanup item.

For the full technical build record — every stage, every test, every number, and the reasoning behind
every non-obvious decision — see [`docs/plan.md`](docs/plan.md). It doubles as the project's
implementation log, not just a prospective plan.

---

## 9. Project structure & docs map

```
fraud-gnn/
├── README.md                  # this file — the audit-level overview
├── LICENSE                    # MIT
├── docs/
│   ├── plan.md                # the full technical build record: every stage, test, and result
│   └── followup.md            # deferred ideas ("future improvements") and things deliberately out of scope
├── notebooks/
│   └── 01_eda.ipynb           # Phase-1 exploratory analysis, with all findings and visualisations
├── reports/                   # exported EDA figures and headline-number summaries
├── data/                      # the Elliptic dataset (not committed — see docs/plan.md to obtain it)
├── src/fraud_gnn/             # all Python source: data loading, model, training, serving, monitoring
│   └── serve/                 # the FastAPI application (/score, /health, /metrics, /feedback)
├── scripts/                   # the Redis loader and the load-test generator
├── tests/                     # the automated test suite (pytest)
├── docker/                    # the API's Dockerfile, Prometheus config, Grafana dashboards
├── docker-compose.yml         # brings up all 4 services together
├── Makefile                   # the commands in §7, spelled out
├── artifacts/                 # trained model + metadata (produced locally by `make train`, not committed)
└── pyproject.toml             # Python dependencies (managed by uv)
```

---

## 10. Credits & license

- **Dataset:** [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
  released by Elliptic and collaborators for anti-money-laundering research on Bitcoin.
- **Code license:** MIT — see [`LICENSE`](LICENSE).
- This is an independent case-study project and is not affiliated with or endorsed by Elliptic.
