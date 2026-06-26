import json, argparse, os
import pandas as pd
import lightgbm as lgb
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

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--stage", type=int, default=1)
args = parser.parse_args()

with open(args.config) as f:
    params = json.load(f)

df       = pd.read_parquet(os.path.expanduser("~/project/data/features_2020_2024.parquet"))
train_df = df[df["timestamp"] <= TRAIN_END]
val_df   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
test_df  = df[df["timestamp"] > VAL_END]

# Stage 1: use only the most recent 15% of training data for fast screening
if args.stage == 1:
    train_df = train_df.tail(int(len(train_df) * 0.15))

model = lgb.LGBMRegressor(**params)
model.fit(
    train_df[FEATURE_COLS], train_df[TARGET_COL],
    eval_set=[(val_df[FEATURE_COLS], val_df[TARGET_COL])],
    callbacks=[lgb.early_stopping(10, verbose=False)],
)

result = {
    "val_mae":  mean_absolute_error(val_df[TARGET_COL], model.predict(val_df[FEATURE_COLS])),
    "test_mae": mean_absolute_error(test_df[TARGET_COL], model.predict(test_df[FEATURE_COLS])),
    "params":   params,
}

with open("/tmp/result.json", "w") as f:
    json.dump(result, f)

print(f"stage={args.stage} val_mae={result['val_mae']:.4f} test_mae={result['test_mae']:.4f}")
