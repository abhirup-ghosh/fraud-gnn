"""FastAPI real-time fraud-scoring service: /score, /health, /metrics."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from fraud_gnn import config as cfg
from fraud_gnn.drift import DriftMonitor
from fraud_gnn.featurestore import get_features, get_redis_client
from fraud_gnn.serve.inference import InferenceEngine
from fraud_gnn.serve.schemas import ScoreRequest, ScoreResponse

fraud_drift_psi = Gauge(
    "fraud_drift_psi", "Mean PSI drift score over the top Phase-1 EDA features"
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = InferenceEngine()
    state["redis_client"] = get_redis_client()
    state["drift_monitor"] = DriftMonitor()
    state["start_time"] = time.time()
    yield
    state.clear()


app = FastAPI(title="fraud-gnn", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    engine: InferenceEngine | None = state.get("engine")
    return {
        "status": "ok",
        "model_version": engine.model_version if engine else "unknown",
        "uptime_s": time.time() - state.get("start_time", time.time()),
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    engine: InferenceEngine = state["engine"]
    start = time.time()

    if req.txId is not None:
        proba = engine.score_known(req.txId, feature_client=state.get("redis_client"))
        raw_features = get_features(req.txId, client=state.get("redis_client"))
        feature_dict = {f"feat_{i + 1}": float(v) for i, v in enumerate(raw_features)}
    else:
        proba = engine.score_inductive(req.features, req.neighbor_features)
        feature_dict = {f"feat_{i + 1}": v for i, v in enumerate(req.features)}

    drift_monitor: DriftMonitor = state["drift_monitor"]
    drift_monitor.add(feature_dict)
    fraud_drift_psi.set(drift_monitor.drift_score)

    label = "illicit" if proba >= engine.threshold else "licit"
    latency_ms = (time.time() - start) * 1000

    return ScoreResponse(
        txId=req.txId,
        illicit_probability=proba,
        label=label,
        threshold=engine.threshold,
        model_version=engine.model_version,
        latency_ms=latency_ms,
        drift_score=drift_monitor.drift_score,
    )
