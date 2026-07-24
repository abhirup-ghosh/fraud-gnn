"""Custom Prometheus metrics: model performance (Layer B) and concept drift (Layer C).

System health (Layer A) comes for free from `prometheus_fastapi_instrumentator`, wired
in app.py.
"""
from collections import OrderedDict, deque

from prometheus_client import Counter, Gauge, Histogram

from fraud_gnn import config as cfg

fraud_predictions_total = Counter(
    "fraud_predictions_total", "Total /score predictions by predicted label", ["label"]
)
fraud_score_histogram = Histogram(
    "fraud_score_histogram",
    "Distribution of predicted illicit probability",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
fraud_flagged_ratio = Gauge(
    "fraud_flagged_ratio",
    f"Rolling share of predictions flagged illicit (last {cfg.FLAGGED_RATIO_WINDOW})",
)
fraud_drift_psi = Gauge(
    "fraud_drift_psi", "Mean PSI drift score over the top Phase-1 EDA features"
)
fraud_rolling_precision = Gauge(
    "fraud_rolling_precision", f"Rolling precision over last {cfg.FEEDBACK_WINDOW} confirmed predictions"
)
fraud_rolling_recall = Gauge(
    "fraud_rolling_recall", f"Rolling recall over last {cfg.FEEDBACK_WINDOW} confirmed predictions"
)
fraud_rolling_f1 = Gauge(
    "fraud_rolling_f1", f"Rolling F1 over last {cfg.FEEDBACK_WINDOW} confirmed predictions"
)


class PredictionTracker:
    """Tracks recent predictions for the flagged-ratio gauge and /feedback correlation."""

    def __init__(self, flagged_window=None, feedback_window=None, cache_size=None):
        self.flagged_window = flagged_window or cfg.FLAGGED_RATIO_WINDOW
        self.feedback_window = feedback_window or cfg.FEEDBACK_WINDOW
        self.cache_size = cache_size or cfg.RECENT_PREDICTIONS_CACHE_SIZE

        self._flagged = deque(maxlen=self.flagged_window)
        self._recent_predictions: OrderedDict[int, bool] = OrderedDict()
        self._confirmed = deque(maxlen=self.feedback_window)

        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def record_prediction(self, is_illicit: bool, tx_id: int | None = None) -> None:
        fraud_predictions_total.labels(label="illicit" if is_illicit else "licit").inc()
        self._flagged.append(is_illicit)
        fraud_flagged_ratio.set(sum(self._flagged) / len(self._flagged))

        if tx_id is not None:
            self._recent_predictions[tx_id] = is_illicit
            if len(self._recent_predictions) > self.cache_size:
                self._recent_predictions.popitem(last=False)

    def record_feedback(self, tx_id: int, true_illicit: bool) -> bool:
        """Returns True if a matching prior prediction was found (and gauges updated)."""
        predicted = self._recent_predictions.get(tx_id)
        if predicted is None:
            return False

        self._confirmed.append((predicted, true_illicit))

        tp = sum(1 for p, a in self._confirmed if p and a)
        fp = sum(1 for p, a in self._confirmed if p and not a)
        fn = sum(1 for p, a in self._confirmed if not p and a)
        self.precision = tp / (tp + fp) if (tp + fp) else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) else 0.0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall)
            else 0.0
        )

        fraud_rolling_precision.set(self.precision)
        fraud_rolling_recall.set(self.recall)
        fraud_rolling_f1.set(self.f1)
        return True
