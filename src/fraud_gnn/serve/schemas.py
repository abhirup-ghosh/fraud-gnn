"""Pydantic request/response models for the /score endpoint."""
from pydantic import BaseModel, model_validator

from fraud_gnn import config as cfg


class ScoreRequest(BaseModel):
    txId: int | None = None
    features: list[float] | None = None
    neighbor_features: list[list[float]] | None = None

    @model_validator(mode="after")
    def check_routing(self):
        if self.txId is None and self.features is None:
            raise ValueError("either txId or features must be provided")
        if self.features is not None and len(self.features) != cfg.N_FEATURES:
            raise ValueError(f"features must have length {cfg.N_FEATURES}")
        if self.neighbor_features is not None:
            for nf in self.neighbor_features:
                if len(nf) != cfg.N_FEATURES:
                    raise ValueError(f"each neighbor_features entry must have length {cfg.N_FEATURES}")
        return self


class ScoreResponse(BaseModel):
    txId: int | None
    illicit_probability: float
    label: str
    threshold: float
    model_version: str
    latency_ms: float
    drift_score: float


class FeedbackRequest(BaseModel):
    txId: int
    true_label: str

    @model_validator(mode="after")
    def check_label(self):
        if self.true_label not in ("illicit", "licit"):
            raise ValueError('true_label must be "illicit" or "licit"')
        return self


class FeedbackResponse(BaseModel):
    matched: bool
    rolling_precision: float | None = None
    rolling_recall: float | None = None
    rolling_f1: float | None = None
