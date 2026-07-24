"""Sends a mix of known-txId and inductive /score requests, ending with a drift-inducing
phase drawn from late/post-shutdown time steps, to exercise the monitoring dashboards."""
import argparse
import random
import statistics
import time

import httpx

from fraud_gnn.data import load_elliptic

DEFAULT_HOST = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--drift-fraction",
        type=float,
        default=0.3,
        help="fraction of requests (at the end of the run) drawn from late/drifted time steps",
    )
    args = parser.parse_args()

    data, txid_to_idx = load_elliptic()
    idx_to_txid = {i: tx for tx, i in txid_to_idx.items()}
    time_step = data.time_step.numpy()
    x = data.x.numpy()

    normal_idx = [i for i in range(len(time_step)) if time_step[i] <= 34]
    drift_idx = [i for i in range(len(time_step)) if time_step[i] >= 43]

    n_drift = int(args.n * args.drift_fraction)
    n_normal = args.n - n_drift

    random.seed(0)
    normal_sample = [random.choice(normal_idx) for _ in range(n_normal)]
    drift_sample = [random.choice(drift_idx) for _ in range(n_drift)]
    plan = [(i, False) for i in normal_sample] + [(i, True) for i in drift_sample]

    latencies = []
    errors = 0
    with httpx.Client(base_url=args.host, timeout=10.0) as client:
        for k, (idx, is_drift_phase) in enumerate(plan):
            tx_id = idx_to_txid[idx]
            use_known = k % 2 == 0
            t0 = time.perf_counter()
            try:
                if use_known:
                    resp = client.post("/score", json={"txId": tx_id})
                else:
                    resp = client.post("/score", json={"features": x[idx].tolist()})
                resp.raise_for_status()
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"request {k} failed: {exc}")
                continue
            latencies.append((time.perf_counter() - t0) * 1000)

            if (k + 1) % 500 == 0:
                phase = "drift" if is_drift_phase else "normal"
                print(f"[{k + 1}/{len(plan)}] phase={phase} last_latency_ms={latencies[-1]:.2f}")

    latencies.sort()
    n = len(latencies)

    def pct(p):
        return latencies[min(int(n * p), n - 1)] if n else float("nan")

    print("\n=== loadgen summary ===")
    print(f"requests sent: {len(plan)}  succeeded: {n}  errors: {errors}")
    if n:
        print(
            f"latency (client-observed round-trip): mean={statistics.mean(latencies):.2f}ms "
            f"p50={pct(0.5):.2f}ms p95={pct(0.95):.2f}ms p99={pct(0.99):.2f}ms max={latencies[-1]:.2f}ms"
        )
        print("p95 < 50ms: PASSED" if pct(0.95) < 50 else "p95 < 50ms: FAILED")


if __name__ == "__main__":
    main()
