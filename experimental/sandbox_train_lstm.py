import json, argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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

# Stage 1: fast screen, Stage 2: medium eval, Stage 3: full training
STAGE_CONFIG = {
    1: {"max_epochs": 5,  "patience": 2},
    2: {"max_epochs": 20, "patience": 5},
    3: {"max_epochs": 50, "patience": 10},
}


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    Xs, ys = [], []
    for i in range(window_size, len(X)):
        Xs.append(X[i - window_size:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Parse args ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--stage", type=int, default=1)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

window_size  = cfg["window_size"]
hidden_size  = cfg["hidden_size"]
num_layers   = cfg["num_layers"]
dropout      = cfg["dropout"]
lr           = cfg["learning_rate"]
batch_size   = cfg["batch_size"]
max_epochs   = STAGE_CONFIG[args.stage]["max_epochs"]
patience     = STAGE_CONFIG[args.stage]["patience"]

# ── Load data ────────────────────────────────────────────────
df = pd.read_parquet(os.path.expanduser("~/project/data/features_2020_2024.parquet"))
train_df = df[df["timestamp"] <= TRAIN_END]
val_df   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
test_df  = df[df["timestamp"] > VAL_END]

# ── Normalize ────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_df[FEATURE_COLS].values)
y_train = scaler_y.fit_transform(train_df[[TARGET_COL]].values).ravel()
X_val   = scaler_X.transform(val_df[FEATURE_COLS].values)
y_val   = scaler_y.transform(val_df[[TARGET_COL]].values).ravel()
X_test  = scaler_X.transform(test_df[FEATURE_COLS].values)
y_test  = scaler_y.transform(test_df[[TARGET_COL]].values).ravel()

# ── Build sequences (include boundary context between splits) ──
all_X = np.vstack([X_train, X_val, X_test])
all_y = np.concatenate([y_train, y_val, y_test])
all_Xs, all_ys = make_sequences(all_X, all_y, window_size)

n_tr = len(X_train) - window_size
n_v  = len(X_val)

X_tr_s, y_tr_s = all_Xs[:n_tr],          all_ys[:n_tr]
X_v_s,  y_v_s  = all_Xs[n_tr:n_tr+n_v],  all_ys[n_tr:n_tr+n_v]
X_te_s, y_te_s = all_Xs[n_tr+n_v:],      all_ys[n_tr+n_v:]

# ── DataLoaders ──────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s)),
    batch_size=batch_size, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_v_s), torch.from_numpy(y_v_s)),
    batch_size=512, shuffle=False,
)

# ── Device (use GPU if available) ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)

# ── Model / optimizer ────────────────────────────────────────
model     = LSTMModel(len(FEATURE_COLS), hidden_size, num_layers, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

# ── Training loop with early stopping ────────────────────────
best_val_loss = float("inf")
best_state    = None
no_improve    = 0

for epoch in range(max_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.cat([
            model(xb.to(device)).cpu() for xb, _ in val_loader
        ]).numpy()
        val_loss = float(np.mean(np.abs(preds - y_v_s)))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

# ── Evaluate with best weights ───────────────────────────────
model.load_state_dict(best_state)
model.to(device)
model.eval()

def predict_mae(X_seq, y_true_scaled):
    with torch.no_grad():
        preds_scaled = model(
            torch.from_numpy(X_seq).to(device)
        ).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    truth = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    return float(mean_absolute_error(truth, preds))

val_mae  = predict_mae(X_v_s,  y_v_s)
test_mae = predict_mae(X_te_s, y_te_s)

result = {"val_mae": val_mae, "test_mae": test_mae, "params": cfg}
with open("/tmp/result.json", "w") as f:
    json.dump(result, f)

print(f"stage={args.stage} val_mae={val_mae:.4f} test_mae={test_mae:.4f}")
