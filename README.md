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

This project implements a **3-stage Hyperband hyperparameter search** for LightGBM electricity price forecasting, using [Daytona](https://daytona.io) to run training jobs in massively parallel sandboxes.

Instead of evaluating hyperparameter configurations one by one (like Optuna), we spin up multiple sandboxes simultaneously — each training a different configuration — and progressively eliminate the worst performers across three stages.

---

## Results

| Method | Trials | Style | Test MAE |
|---|---|---|---|
| Optuna (baseline) | 30 | Sequential | 7.2300 EUR/MWh |
| **Hyperband + Daytona** | **36** | **Parallel** | **7.1754 EUR/MWh** ✅ |

**Improvement: +0.055 EUR/MWh (0.76% better than baseline)**

---

## Architecture

```
Stage 1 — Fast Screen (10% of training data)
  4 batches × 9 sandboxes = 36 configs explored in parallel
  → Keep Top 9

Stage 2 — Medium Evaluation (33% of training data)
  10 sandboxes in parallel (Top 9 from Stage 1 + 1 seed)
  → Keep Top 5

Stage 3 — Full Training (100% of training data)
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
├── orchestrator.py      # Main Hyperband loop — controls all sandboxes
├── sandbox_train.py     # Training script that runs inside each sandbox
├── setup_snapshot.py    # One-time setup: creates the Daytona snapshot
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
1. Spawns 9 sandboxes in parallel for each Stage 1 batch
2. Each sandbox clones the repo, writes its config, and runs `sandbox_train.py`
3. Results are collected, sorted by `val_mae`, and the worst configs are eliminated
4. Survivors advance to the next stage with more training data

### 3. Stage-aware early stopping

| Stage | Data | Early Stopping Rounds |
|---|---|---|
| Stage 1 | 10% | 10 |
| Stage 2 | 33% | 30 |
| Stage 3 | 100% | 100 |

---

## Requirements

```bash
pip install daytona lightgbm pandas scikit-learn pyarrow numpy
```

Set your Daytona API key:

```bash
export DAYTONA_API_KEY="your-api-key"
```

---

## Daytona Features Used

| Feature | Usage |
|---|---|
| **Sandbox from Snapshot** | Pre-installed packages, instant startup |
| **Parallel sandboxes** | Up to 9 simultaneous training jobs |
| **`process.code_run()`** | Write config safely without shell escaping issues |
| **`process.exec()`** | Run training script and collect results |
| **Auto cleanup** | `sb.delete()` in `finally` block — no wasted resources |

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
  "reg_lambda": 1.96
}
```

**Test MAE: 7.1754 EUR/MWh**
