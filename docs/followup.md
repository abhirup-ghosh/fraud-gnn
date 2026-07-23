# Follow-up

Two running lists maintained throughout the project. Add to them as you go — do **not** act on them
during the current stage.

---

## 1. Future improvements

Ideas we recognise as worth doing but deliberately defer (we note them at the stage where the idea
arises, then keep building). Each entry: what, why it helps, and which `plan.md` stage it relates to.

| Idea | Why it helps | Related plan stage | Added on |
|------|--------------|--------------------|----------|
| _(none yet)_ | | | |

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
