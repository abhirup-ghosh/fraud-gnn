"""Single source of truth for paths, splits, hyperparameters, and monitoring constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FRAUD_GNN_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("FRAUD_GNN_ARTIFACTS_DIR", ROOT / "artifacts"))

CLASSES_CSV = DATA_DIR / "elliptic_txs_classes.csv"
EDGELIST_CSV = DATA_DIR / "elliptic_txs_edgelist.csv"
FEATURES_CSV = DATA_DIR / "elliptic_txs_features.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.pt"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
TXID_TO_IDX_PATH = ARTIFACTS_DIR / "txid_to_idx.json"
REFERENCE_STATS_PATH = ARTIFACTS_DIR / "reference_stats.parquet"

SEED = 42

N_NODES = 203_769
N_EDGES = 234_355
N_FEATURES = 165
N_LOCAL_FEATURES = 93  # feat_1..feat_93
N_AGG_FEATURES = N_FEATURES - N_LOCAL_FEATURES  # feat_94..feat_165

# Temporal split (§0 ground truth: 100% of edges are intra-time-step)
TRAIN_MAX_STEP = 34
TEST_MIN_STEP = 35
VAL_FRACTION = 0.15  # random stratified slice carved out of train, NOT temporal

# GraphSAGE hyperparameters (§S2)
HIDDEN = 128
DROPOUT = 0.4
LR = 5e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 300
PATIENCE = 30
AGGR = "mean"
OUT_DIM = 2

# RandomForest baseline (§S3)
RF_N_ESTIMATORS = 200

# Top features by |Cohen's d| from Phase-1 EDA — used for drift monitoring (§S6)
TOP_DRIFT_FEATURES = ["feat_53", "feat_55", "feat_89", "feat_90", "feat_91", "feat_52"]

# Drift monitor (§S6)
DRIFT_BUFFER = 1000
DRIFT_WINDOW = 200
DRIFT_PSI_WARN = 0.1
DRIFT_PSI_ALERT = 0.25

# Serving (§S8)
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
DEFAULT_THRESHOLD = 0.5
