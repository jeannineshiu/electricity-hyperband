"""
Model registry — single source of truth for all model metadata.

Contains:
  - MODEL_DEFAULTS : fixed params injected into every sampled config
  - MODEL_BOUNDS   : valid ranges used by the LLM agent for validation
  - Param samplers  : random search spaces for each model
  - Seeds           : known-good LightGBM configs from previous runs
"""

import random
import config   # reads config.MODEL_TYPE at call time (not at import time)

# ── Model defaults (shared with agent/ml_engineer.py) ────────
MODEL_DEFAULTS = {
    "lightgbm": {"random_state": 42, "n_jobs": -1, "verbose": -1},
    "xgboost":  {"random_state": 42, "n_jobs": -1, "verbosity": 0},
    "catboost": {"random_seed": 42, "bootstrap_type": "Bernoulli"},
    "rf":       {"random_state": 42, "n_jobs": -1},
}

# ── Param bounds (used by LLM agent for validation) ──────────
MODEL_BOUNDS = {
    "lightgbm": {
        "n_estimators": (100, 10000), "max_depth": (2, 12),
        "learning_rate": (0.001, 0.5), "num_leaves": (5, 300),
        "subsample": (0.3, 1.0), "colsample_bytree": (0.3, 1.0),
        "min_child_samples": (1, 500),
        "reg_alpha": (0.0, 20.0), "reg_lambda": (0.0, 20.0),
    },
    "xgboost": {
        "n_estimators": (100, 5000), "max_depth": (2, 12),
        "learning_rate": (0.001, 0.5), "subsample": (0.3, 1.0),
        "colsample_bytree": (0.3, 1.0), "min_child_weight": (1, 50),
        "gamma": (0.0, 5.0), "reg_alpha": (0.0, 20.0), "reg_lambda": (0.0, 20.0),
    },
    "catboost": {
        "iterations": (100, 5000), "depth": (2, 12),
        "learning_rate": (0.001, 0.5), "l2_leaf_reg": (0.1, 50.0),
        "subsample": (0.3, 1.0), "rsm": (0.3, 1.0),
    },
    "rf": {
        "n_estimators": (50, 2000), "max_depth": (2, 50),
        "min_samples_split": (2, 50), "min_samples_leaf": (1, 50),
    },
}

# ── Param samplers ────────────────────────────────────────────
def _sample_lightgbm() -> dict:
    return {
        **MODEL_DEFAULTS["lightgbm"],
        "n_estimators":      random.choice([1000, 1500, 2000, 3000, 5000]),
        "max_depth":         random.randint(3, 6),
        "learning_rate":     random.choice([0.003, 0.005, 0.01, 0.02, 0.03]),
        "num_leaves":        random.randint(15, 63),
        "subsample":         round(random.uniform(0.6, 0.9), 2),
        "colsample_bytree":  round(random.uniform(0.6, 0.9), 2),
        "min_child_samples": random.randint(20, 100),
        "reg_alpha":         round(random.uniform(0.0, 2.0), 2),
        "reg_lambda":        round(random.uniform(0.1, 2.0), 2),
    }

def _sample_xgboost() -> dict:
    return {
        **MODEL_DEFAULTS["xgboost"],
        "n_estimators":     random.choice([200, 400, 600, 800, 1000]),
        "max_depth":        random.randint(3, 8),
        "learning_rate":    random.choice([0.01, 0.05, 0.1, 0.2]),
        "subsample":        round(random.uniform(0.6, 1.0), 2),
        "colsample_bytree": round(random.uniform(0.6, 1.0), 2),
        "min_child_weight": random.randint(1, 10),
        "reg_alpha":        round(random.uniform(0.0, 1.0), 2),
        "reg_lambda":       round(random.uniform(0.1, 2.0), 2),
        "gamma":            round(random.uniform(0.0, 0.5), 2),
    }

def _sample_catboost() -> dict:
    return {
        **MODEL_DEFAULTS["catboost"],
        "iterations":    random.choice([200, 400, 600, 800]),
        "depth":         random.randint(4, 10),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "l2_leaf_reg":   round(random.uniform(1.0, 10.0), 1),
        "subsample":     round(random.uniform(0.6, 1.0), 2),
        "rsm":           round(random.uniform(0.6, 1.0), 2),
    }

def _sample_rf() -> dict:
    return {
        **MODEL_DEFAULTS["rf"],
        "n_estimators":      random.choice([100, 200, 300, 500]),
        "max_depth":         random.choice([5, 10, 15, 20, None]),
        "min_samples_split": random.randint(2, 20),
        "min_samples_leaf":  random.randint(1, 10),
        "max_features":      random.choice(["sqrt", "log2", 0.5, 0.7]),
    }

_PARAM_SAMPLERS = {
    "lightgbm": _sample_lightgbm,
    "xgboost":  _sample_xgboost,
    "catboost": _sample_catboost,
    "rf":       _sample_rf,
}

def sample_params() -> dict:
    """Sample hyperparameters for the current config.MODEL_TYPE."""
    return _PARAM_SAMPLERS[config.MODEL_TYPE]()

# ── Known-good seeds (LightGBM only) ─────────────────────────
_BEST_LGB = {
    "n_estimators": 5000, "max_depth": 5, "learning_rate": 0.02,
    "num_leaves": 47, "subsample": 0.86, "colsample_bytree": 0.89,
    "min_child_samples": 36, "reg_alpha": 1.26, "reg_lambda": 1.96,
    **MODEL_DEFAULTS["lightgbm"],
}
_LGB_SEEDS = [
    {**_BEST_LGB, "random_state": 42},
    {**_BEST_LGB, "random_state": 123},
    {**_BEST_LGB, "random_state": 456},
    {**_BEST_LGB, "random_state": 789},
    {**_BEST_LGB, "learning_rate": 0.01, "random_state": 42},
]

def get_seed_params() -> list:
    """Returns known-good seeds for the current MODEL_TYPE (LightGBM only)."""
    return _LGB_SEEDS if config.MODEL_TYPE == "lightgbm" else []
