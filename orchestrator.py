"""
Orchestrator — CLI entry point for the 3-stage Hyperband search.

Thin layer that composes:
  config           → search parameters
  models.registry  → param sampling and seeds
  hyperband        → parallel sandbox execution
  tracking         → MLflow logging
  agent.history    → persistent run memory
"""

import json
import time
from datetime import datetime

import mlflow

import config
from models.registry import sample_params, get_seed_params
from hyperband import stream_stage
from tracking.mlflow_logger import start_experiment, log_trial, log_hyperband_summary


def run_parallel(configs: list, stage: int, batch: int = None) -> list:
    """Runs stream_stage and logs each result to MLflow."""
    results = []
    for r in stream_stage(configs, stage):
        results.append(r)
        print(f"    stage={stage} val_mae={r['val_mae']:.4f}")
        log_trial(r["params"], r["val_mae"], r["test_mae"], stage, batch)
    return results


def run_hyperband():
    start_experiment("electricity-hyperband")

    with mlflow.start_run(run_name="hyperband_search"):
        mlflow.log_param("model_type", config.MODEL_TYPE)
        mlflow.log_param("n_batches",  config.N_BATCHES)
        mlflow.log_param("n_batch",    config.N_BATCH)
        mlflow.log_param("top_s2",     config.TOP_S2)
        mlflow.log_param("top_s3",     config.TOP_S3)
        mlflow.log_param("baseline",   config.BASELINE)

        # ── Stage 1 ───────────────────────────────────────────
        t_start = time.time()
        n_total = config.N_BATCH * config.N_BATCHES
        print(f"Stage 1: {config.N_BATCHES} batches × {config.N_BATCH} sandboxes × 10% data  ({n_total} configs total)")
        all_s1 = []
        for b in range(config.N_BATCHES):
            print(f"  Batch {b + 1}/{config.N_BATCHES}:")
            configs = [sample_params() for _ in range(config.N_BATCH)]
            all_s1.extend(run_parallel(configs, stage=1, batch=b + 1))

        if not all_s1:
            raise RuntimeError("All Stage 1 sandboxes failed.")
        all_s1.sort(key=lambda x: x["val_mae"])
        survivors_s2 = all_s1[:config.TOP_S2]
        seeds = get_seed_params()
        for seed in seeds:
            survivors_s2.append({"val_mae": 0.0, "test_mae": 0.0, "params": seed})
        print(f"\nStage 1 → Top {config.TOP_S2} + {len(seeds)} seeds → Stage 2 ({len(survivors_s2)} total)")

        # ── Stage 2 ───────────────────────────────────────────
        print(f"\nStage 2: {len(survivors_s2)} sandboxes × 33% data")
        s2_results = run_parallel([r["params"] for r in survivors_s2], stage=2)
        if not s2_results:
            raise RuntimeError("All Stage 2 sandboxes failed.")
        s2_results.sort(key=lambda x: x["val_mae"])
        survivors_s3 = s2_results[:config.TOP_S3]
        print(f"\nStage 2 → Top {config.TOP_S3} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s3]}")

        # ── Stage 3 ───────────────────────────────────────────
        print(f"\nStage 3: {len(survivors_s3)} sandboxes × 100% data")
        s3_results = run_parallel([r["params"] for r in survivors_s3], stage=3)
        if not s3_results:
            raise RuntimeError("All Stage 3 sandboxes failed.")
        best = min(s3_results, key=lambda x: x["test_mae"])

        elapsed = time.time() - t_start
        mins, secs = divmod(int(elapsed), 60)

        log_hyperband_summary(
            best_params=best["params"],
            best_test_mae=best["test_mae"],
            baseline=config.BASELINE,
            n_total=n_total,
            elapsed_sec=int(elapsed),
        )

    print(f"\n{'='*50}")
    print(f"Total wall-clock time                    : {mins}m {secs}s")
    print(f"Total configs explored                   : {n_total}")
    print(f"Baseline (Optuna 30 trials, sequential)  : {config.BASELINE}")
    print(f"Hyperband 3-stage ({config.N_BATCHES}×{config.N_BATCH}→{config.TOP_S2}→{config.TOP_S3}) : {best['test_mae']:.4f}")
    print(f"Delta                                    : {config.BASELINE - best['test_mae']:+.4f} EUR/MWh")
    print(f"\nBest params (seed for next run):")
    print(json.dumps(best["params"], indent=2))
    print(f"\nMLflow UI: run `mlflow ui --port 5001` then open http://localhost:5001")

    try:
        from agent.history import save as save_history
        save_history({
            "timestamp":     datetime.now().isoformat(),
            "model_type":    config.MODEL_TYPE,
            "n_trials":      n_total,
            "reasoning":     "CLI hyperband run",
            "search_space":  "orchestrator default",
            "best_test_mae": best["test_mae"],
            "best_val_mae":  best["val_mae"],
            "best_params":   best["params"],
        })
    except Exception:
        pass


if __name__ == "__main__":
    run_hyperband()
