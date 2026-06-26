# Electricity Price Forecasting — Parallel Hyperband with Daytona

> Beat a sequential Optuna baseline (MAE 7.23) using parallel Hyperband search across Daytona sandboxes.
> **Final result: MAE 7.1754 EUR/MWh** (+0.055 improvement)

---

## Inspiration

This project is built upon ideas and baselines from a prior MLOps project:
**[electricity-price-forecasting](https://github.com/jeannineshiu/electricity-price-forecasting)**

That project established the ENTSO-E data pipeline, LightGBM feature engineering, and the Optuna sequential search baseline (MAE 7.23) that this Hyperband implementation aims to surpass.

---

## Overview

This project implements a **3-stage Hyperband hyperparameter search** for LightGBM electricity price forecasting, using [Daytona](https://daytona.io) to run training jobs in massively parallel sandboxes, with **MLflow** for full experiment tracking and observability.

Instead of evaluating hyperparameter configurations one by one (like Optuna), we spin up multiple sandboxes simultaneously — each training a different configuration — and progressively eliminate the worst performers across three stages.

---

## Results

| Method | Trials | Style | Test MAE |
|---|---|---|---|
| Optuna (baseline) | 30 | Sequential | 7.2300 EUR/MWh |
| **Hyperband + Daytona** | **36** | **Parallel** | **7.1754 EUR/MWh** ✅ |

**Improvement: +0.055 EUR/MWh (0.76% better than baseline)**

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   orchestrator.py                   │
│              (runs on local machine)                │
│                                                     │
│  ThreadPoolExecutor → 9 Daytona sandboxes at once  │
│                       ↓                            │
│              MLflow tracking (localhost)            │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   [Sandbox 1]   [Sandbox 2]  ... [Sandbox 9]
   git clone      git clone        git clone
   sandbox_train  sandbox_train    sandbox_train
   → result.json  → result.json   → result.json
```

### 3-Stage Hyperband

```
Stage 1 — Fast Screen (10% of training data, early_stop=10)
  4 batches × 9 sandboxes = 36 configs explored in parallel
  → Keep Top 5

Stage 2 — Medium Evaluation (33% of training data, early_stop=30)
  10 sandboxes in parallel (Top 5 from Stage 1 + 5 seeds)
  → Keep Top 5

Stage 3 — Full Training (100% of training data, early_stop=100)
  5 sandboxes in parallel
  → Best config wins
```

Each stage uses progressively more data. Configs that perform poorly are eliminated early, saving ~70% of compute compared to training all 36 on full data.

---

## Dataset

- **Source**: ENTSO-E European day-ahead electricity prices
- **File**: `data/features_2020_2024.parquet` (43,680 hourly rows)
- **Features**: lag features (1h, 24h, 168h), rolling statistics, calendar features
- **Split**:
  - Train: 2020–2022
  - Validation: 2023
  - Test: 2024

---

## Hyperparameter Search Space

| Parameter | Range |
|---|---|
| `n_estimators` | 1000, 1500, 2000, 3000, 5000 |
| `max_depth` | 3–6 |
| `learning_rate` | 0.003, 0.005, 0.01, 0.02, 0.03 |
| `num_leaves` | 15–63 |
| `subsample` | 0.6–0.9 |
| `colsample_bytree` | 0.6–0.9 |
| `min_child_samples` | 20–100 |
| `reg_alpha` | 0.0–2.0 |
| `reg_lambda` | 0.1–2.0 |

---

## Project Structure

```
electricity-hyperband/
├── orchestrator.py           # Main Hyperband loop — controls all sandboxes
├── orchestrator_lstm.py      # LSTM variant of Hyperband
├── sandbox_train.py          # LightGBM training script (runs inside sandbox)
├── sandbox_train_lstm.py     # LSTM training script (runs inside sandbox)
├── setup_snapshot.py         # One-time setup: LightGBM Daytona snapshot
├── setup_snapshot_lstm.py    # One-time setup: LSTM Daytona snapshot
├── tracking/
│   ├── __init__.py
│   └── mlflow_logger.py      # MLflow experiment tracking helpers
└── data/
    └── features_2020_2024.parquet
```

---

## How It Works

### 1. Setup (run once)

```bash
python setup_snapshot.py
```

Creates a Daytona snapshot (`elec-forecast-v2`) with all Python packages pre-installed (`lightgbm`, `pandas`, `scikit-learn`, `pyarrow`, `numpy`). All subsequent sandboxes boot from this snapshot instantly.

### 2. Run the Hyperband search

```bash
python orchestrator.py
```

The orchestrator:
1. Starts an MLflow parent run (`hyperband_search`)
2. Spawns 9 sandboxes in parallel for each Stage 1 batch
3. Each sandbox clones the repo, writes its config, and runs `sandbox_train.py`
4. Results are collected, logged to MLflow, sorted by `val_mae`, and worst configs are eliminated
5. Survivors advance to the next stage with more training data

### 3. View experiment results in MLflow

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

Each Hyperband run creates:
- A **parent run** with search configuration and final best metrics
- **Nested trial runs** for every sandbox, each with full hyperparameters and stage-level MAE

### 4. Real-time Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## Requirements

```bash
pip install daytona lightgbm pandas scikit-learn pyarrow numpy mlflow
```

Set your Daytona API key:

```bash
export DAYTONA_API_KEY="your-api-key"
```

---

## Daytona Features Used

| Feature | Usage |
|---|---|
| **Sandbox from Snapshot** | Pre-installed packages, instant startup — setup cost paid once |
| **Parallel sandboxes** | Up to 9 simultaneous training jobs per batch |
| **`process.code_run()`** | Write config safely without shell escaping issues |
| **`process.exec()`** | Run training script and collect results |
| **Auto cleanup** | `sb.delete()` in `finally` block — no idle costs |
| **Fault isolation** | One sandbox crash does not affect others — `[SKIP]` pattern |

---

## Real-time Dashboard

Click **▶ Start** to launch the search. The leaderboard updates live as each sandbox completes.

| Start | Running | Complete |
|:---:|:---:|:---:|
| ![Dashboard start](docs/screenshots/dashboard_01.png) | ![Dashboard running](docs/screenshots/dashboard_02.png) | ![Dashboard complete](docs/screenshots/dashboard_03.png) |

---

## MLflow Tracking

Every Hyperband run is fully logged to MLflow:

| What is tracked | Where |
|---|---|
| Search config (`n_batches`, `top_s2`, `top_s3`, `baseline`) | Parent run — params |
| Each trial's full hyperparameters | Nested run — params |
| `val_mae` and `test_mae` per trial | Nested run — metrics |
| Stage and batch tags | Nested run — tags |
| Best params, best MAE, improvement, wall-clock time | Parent run — metrics |

Run names follow `s{stage}_b{batch}_trial_{n}` format (e.g. `s1_b2_trial_015`) for easy navigation in the UI.

![MLflow UI](docs/screenshots/mlflow_ui.png)

---

## Why Daytona?

| Property | What it means in practice |
|---|---|
| **Isolation** | Each sandbox is an independent environment — no package conflicts, one crash doesn't cascade |
| **Snapshot** | Environment setup cost is paid once, not once per trial |
| **Ephemeral Compute** | `sb.delete()` after training — no idle servers, pay for seconds used |
| **Reproducibility** | Same snapshot + same code = identical environment, reproducible results |
| **Scalability** | Change `N_BATCH = 9` to `N_BATCH = 90` — orchestrator code stays the same |
| **Fault Tolerance** | Failed sandboxes are skipped, search continues with remaining results |

---

## Best Found Configuration

```json
{
  "n_estimators": 5000,
  "max_depth": 5,
  "learning_rate": 0.02,
  "num_leaves": 47,
  "subsample": 0.86,
  "colsample_bytree": 0.89,
  "min_child_samples": 36,
  "reg_alpha": 1.26,
  "reg_lambda": 1.96,
  "random_state": 42
}
```

**Test MAE: 7.1754 EUR/MWh**

---

## Roadmap

- [x] 3-stage Hyperband with Daytona parallel sandboxes
- [x] MLflow experiment tracking
- [ ] Real-time dashboard (Streamlit)
- [ ] Generic model interface (LightGBM / XGBoost / CatBoost / Random Forest)
- [ ] LLM-powered ML agent (diagnose → suggest → re-run)
- [ ] Full forecasting platform (upload CSV → auto HPO → deploy API → monitor drift)
