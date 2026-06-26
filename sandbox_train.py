import json, argparse, os
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
]
TARGET_COL = "price_eur_mwh"
TRAIN_END  = "2022-12-31 23:00:00+00:00"
VAL_END    = "2023-12-31 23:00:00+00:00"

# Stage 1: 10% data for fast screening
# Stage 2: 33% data for medium evaluation
# Stage 3: 100% data for final training
STAGE_EARLY_STOP = {1: 10, 2: 30, 3: 100}


def fit_lightgbm(params, X_tr, y_tr, X_v, y_v, early_stop):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )
    return model


def fit_xgboost(params, X_tr, y_tr, X_v, y_v, early_stop):
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        early_stopping_rounds=early_stop,
        verbose=False,
    )
    return model


def fit_catboost(params, X_tr, y_tr, X_v, y_v, early_stop):
    model = CatBoostRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_v, y_v),
        early_stopping_rounds=early_stop,
        verbose=False,
    )
    return model


def fit_rf(params, X_tr, y_tr, X_v, y_v, early_stop):
    model = RandomForestRegressor(**params)
    model.fit(X_tr, y_tr)  # RF does not support early stopping
    return model


MODEL_FITTERS = {
    "lightgbm": fit_lightgbm,
    "xgboost":  fit_xgboost,
    "catboost": fit_catboost,
    "rf":       fit_rf,
}

# ── Parse args ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--stage", type=int, default=1)
parser.add_argument("--model", default="lightgbm", choices=list(MODEL_FITTERS.keys()))
args = parser.parse_args()

with open(args.config) as f:
    params = json.load(f)

early_stop = STAGE_EARLY_STOP[args.stage]

# ── Load & split data ─────────────────────────────────────────
df = pd.read_parquet(os.path.expanduser("~/project/data/features_2020_2024.parquet"))
train_df = df[df["timestamp"] <= TRAIN_END]
val_df   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
test_df  = df[df["timestamp"] > VAL_END]

if args.stage == 1:
    train_df = train_df.tail(int(len(train_df) * 0.10))
elif args.stage == 2:
    train_df = train_df.tail(int(len(train_df) * 0.33))

X_tr = train_df[FEATURE_COLS].values
y_tr = train_df[TARGET_COL].values
X_v  = val_df[FEATURE_COLS].values
y_v  = val_df[TARGET_COL].values
X_te = test_df[FEATURE_COLS].values
y_te = test_df[TARGET_COL].values

# ── Train ─────────────────────────────────────────────────────
fitter = MODEL_FITTERS[args.model]
model  = fitter(params, X_tr, y_tr, X_v, y_v, early_stop)

# ── Evaluate ──────────────────────────────────────────────────
result = {
    "val_mae":  float(mean_absolute_error(y_v,  model.predict(X_v))),
    "test_mae": float(mean_absolute_error(y_te, model.predict(X_te))),
    "params":   params,
    "model":    args.model,
}

with open("/tmp/result.json", "w") as f:
    json.dump(result, f)

print(f"stage={args.stage} model={args.model} "
      f"val_mae={result['val_mae']:.4f} test_mae={result['test_mae']:.4f}")
