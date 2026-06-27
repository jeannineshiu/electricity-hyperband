"""
Project-wide configuration constants.

MODEL_TYPE is the only mutable variable — updated at runtime by the dashboard
and the agent. All other values are fixed for the project lifetime.
"""

SNAPSHOT   = "elec-forecast-v3"
REPO_URL   = "https://github.com/jeannineshiu/electricity-hyperband"

MODEL_TYPE = "lightgbm"   # mutable: "lightgbm" | "xgboost" | "catboost" | "rf"

N_BATCH    = 9   # sandboxes per Stage-1 batch (Daytona concurrency limit is 10)
N_BATCHES  = 4   # Stage-1 batches → 36 configs total
TOP_S2     = 5   # top configs advancing to Stage 2
TOP_S3     = 5   # top configs advancing to Stage 3

BASELINE   = 7.23   # Optuna 30-trial sequential baseline (EUR/MWh)
