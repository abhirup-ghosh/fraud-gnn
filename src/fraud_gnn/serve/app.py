"""FastAPI real-time fraud-scoring service: /score, /health, /metrics, /feedback."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from fraud_gnn.drift import DriftMonitor
from fraud_gnn.featurestore import get_features, get_redis_client
from fraud_gnn.serve.inference import InferenceEngine
from fraud_gnn.serve.metrics import (
    PredictionTracker,
    fraud_drift_psi,
    fraud_score_histogram,
)
from fraud_gnn.serve.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    ScoreRequest,
    ScoreResponse,
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = InferenceEngine()
    state["redis_client"] = get_redis_client()
    state["drift_monitor"] = DriftMonitor()
    state["tracker"] = PredictionTracker()
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

    is_illicit = proba >= engine.threshold
    label = "illicit" if is_illicit else "licit"
    latency_ms = (time.time() - start) * 1000

    fraud_score_histogram.observe(proba)
    tracker: PredictionTracker = state["tracker"]
    tracker.record_prediction(is_illicit, tx_id=req.txId)

    return ScoreResponse(
        txId=req.txId,
        illicit_probability=proba,
        label=label,
        threshold=engine.threshold,
        model_version=engine.model_version,
        latency_ms=latency_ms,
        drift_score=drift_monitor.drift_score,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    tracker: PredictionTracker = state["tracker"]
    matched = tracker.record_feedback(req.txId, req.true_label == "illicit")
    if not matched:
        return FeedbackResponse(matched=False)
    return FeedbackResponse(
        matched=True,
        rolling_precision=tracker.precision,
        rolling_recall=tracker.recall,
        rolling_f1=tracker.f1,
    )
