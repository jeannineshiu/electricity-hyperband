import json, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from daytona import Daytona, CreateSandboxFromSnapshotParams
import mlflow
from tracking.mlflow_logger import start_experiment, log_trial, log_hyperband_summary

SNAPSHOT   = "elec-forecast-v3"
REPO_URL   = "https://github.com/jeannineshiu/electricity-hyperband"
MODEL_TYPE = "lightgbm"   # switch to "xgboost", "catboost", or "rf"
N_BATCH    = 9   # sandboxes per batch (within the 10-sandbox concurrency limit)
N_BATCHES  = 4   # Stage 1 batches → 36 configs total
TOP_S2     = 5   # top configs advancing to Stage 2
TOP_S3     = 5   # top configs advancing to Stage 3
BASELINE   = 7.23

# ── Model-specific param samplers ─────────────────────────────
def _sample_lightgbm():
    return {
        "n_estimators":      random.choice([1000, 1500, 2000, 3000, 5000]),
        "max_depth":         random.randint(3, 6),
        "learning_rate":     random.choice([0.003, 0.005, 0.01, 0.02, 0.03]),
        "num_leaves":        random.randint(15, 63),
        "subsample":         round(random.uniform(0.6, 0.9), 2),
        "colsample_bytree":  round(random.uniform(0.6, 0.9), 2),
        "min_child_samples": random.randint(20, 100),
        "reg_alpha":         round(random.uniform(0.0, 2.0), 2),
        "reg_lambda":        round(random.uniform(0.1, 2.0), 2),
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }

def _sample_xgboost():
    return {
        "n_estimators":     random.choice([200, 400, 600, 800, 1000]),
        "max_depth":        random.randint(3, 8),
        "learning_rate":    random.choice([0.01, 0.05, 0.1, 0.2]),
        "subsample":        round(random.uniform(0.6, 1.0), 2),
        "colsample_bytree": round(random.uniform(0.6, 1.0), 2),
        "min_child_weight": random.randint(1, 10),
        "reg_alpha":        round(random.uniform(0.0, 1.0), 2),
        "reg_lambda":       round(random.uniform(0.1, 2.0), 2),
        "gamma":            round(random.uniform(0.0, 0.5), 2),
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    }

def _sample_catboost():
    return {
        "iterations":     random.choice([200, 400, 600, 800]),
        "depth":          random.randint(4, 10),
        "learning_rate":  random.choice([0.01, 0.05, 0.1, 0.2]),
        "l2_leaf_reg":    round(random.uniform(1.0, 10.0), 1),
        "bootstrap_type": "Bernoulli",          # required for subsample to work
        "subsample":      round(random.uniform(0.6, 1.0), 2),
        "rsm":            round(random.uniform(0.6, 1.0), 2),
        "random_seed": 42,
    }

def _sample_rf():
    return {
        "n_estimators":      random.choice([100, 200, 300, 500]),
        "max_depth":         random.choice([5, 10, 15, 20, None]),
        "min_samples_split": random.randint(2, 20),
        "min_samples_leaf":  random.randint(1, 10),
        "max_features":      random.choice(["sqrt", "log2", 0.5, 0.7]),
        "random_state": 42, "n_jobs": -1,
    }

_PARAM_SAMPLERS = {
    "lightgbm": _sample_lightgbm,
    "xgboost":  _sample_xgboost,
    "catboost": _sample_catboost,
    "rf":       _sample_rf,
}

def sample_params():
    return _PARAM_SAMPLERS[MODEL_TYPE]()

# Known good LightGBM seeds — evaluated at call time so MODEL_TYPE switching works
_BEST_LGB = {
    "n_estimators": 5000, "max_depth": 5, "learning_rate": 0.02,
    "num_leaves": 47, "subsample": 0.86, "colsample_bytree": 0.89,
    "min_child_samples": 36, "reg_alpha": 1.26, "reg_lambda": 1.96,
    "n_jobs": -1, "verbose": -1,
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
    return _LGB_SEEDS if MODEL_TYPE == "lightgbm" else []

daytona = Daytona()


def run_sandbox(params, stage):
    sb = daytona.create(CreateSandboxFromSnapshotParams(snapshot=SNAPSHOT))
    try:
        clone_resp = sb.process.exec(f"git clone {REPO_URL} $HOME/project")
        if clone_resp.exit_code != 0:
            raise RuntimeError(f"git clone failed: {clone_resp.result}")

        # Write config via code_run to avoid shell escaping issues with special characters
        sb.process.code_run(f"""
import json
with open('/tmp/config.json', 'w') as f:
    json.dump({params}, f)
""")
        train_resp = sb.process.exec(
            f"python $HOME/project/sandbox_train.py "
            f"--config /tmp/config.json --stage {stage} --model {MODEL_TYPE}"
        )
        if train_resp.exit_code != 0:
            raise RuntimeError(f"Training failed (exit {train_resp.exit_code}): {train_resp.result}")
        resp = sb.process.exec("cat /tmp/result.json")
        if not resp.result or not resp.result.strip():
            raise RuntimeError("result.json is empty")
        return json.loads(resp.result)
    finally:
        sb.delete()


def stream_stage(configs: list, stage: int):
    """Generator: yields one result dict per sandbox as it completes.
    No MLflow logging — used by dashboard for real-time UI updates."""
    with ThreadPoolExecutor(max_workers=max(1, len(configs))) as ex:
        futures = {ex.submit(run_sandbox, cfg, stage): cfg for cfg in configs}
        for f in as_completed(futures):
            try:
                yield f.result()
            except Exception as e:
                print(f"    [SKIP] {e}", flush=True)


def run_parallel(configs, stage, batch=None):
    """CLI wrapper: calls stream_stage and logs each result to MLflow."""
    results = []
    for r in stream_stage(configs, stage):
        results.append(r)
        print(f"    stage={stage} val_mae={r['val_mae']:.4f}")
        log_trial(r["params"], r["val_mae"], r["test_mae"], stage, batch)
    return results


def run_hyperband():
    start_experiment("electricity-hyperband")

    with mlflow.start_run(run_name="hyperband_search"):
        mlflow.log_param("n_batches", N_BATCHES)
        mlflow.log_param("n_batch",   N_BATCH)
        mlflow.log_param("top_s2",    TOP_S2)
        mlflow.log_param("top_s3",    TOP_S3)
        mlflow.log_param("baseline",  BASELINE)

        # ── Stage 1 ───────────────────────────────────────────
        t_start = time.time()
        n_total = N_BATCH * N_BATCHES
        print(f"Stage 1: {N_BATCHES} batches × {N_BATCH} sandboxes × 10% data  ({n_total} configs total)")
        all_s1 = []
        for b in range(N_BATCHES):
            print(f"  Batch {b + 1}/{N_BATCHES}:")
            configs = [sample_params() for _ in range(N_BATCH)]
            all_s1.extend(run_parallel(configs, stage=1, batch=b + 1))

        if not all_s1:
            raise RuntimeError("All Stage 1 sandboxes failed.")
        all_s1.sort(key=lambda x: x["val_mae"])
        survivors_s2 = all_s1[:TOP_S2]
        seeds = get_seed_params()
        for seed in seeds:
            survivors_s2.append({"val_mae": 0.0, "test_mae": 0.0, "params": seed})
        print(f"\nStage 1 → Top {TOP_S2} + {len(seeds)} seeds → Stage 2 ({len(survivors_s2)} total)")

        # ── Stage 2 ───────────────────────────────────────────
        print(f"\nStage 2: {len(survivors_s2)} sandboxes × 33% data")
        s2_results = run_parallel([r["params"] for r in survivors_s2], stage=2)
        if not s2_results:
            raise RuntimeError("All Stage 2 sandboxes failed.")
        s2_results.sort(key=lambda x: x["val_mae"])
        survivors_s3 = s2_results[:TOP_S3]
        print(f"\nStage 2 → Top {TOP_S3} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s3]}")

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
            baseline=BASELINE,
            n_total=n_total,
            elapsed_sec=int(elapsed),
        )

    print(f"\n{'='*50}")
    print(f"Total wall-clock time                    : {mins}m {secs}s")
    print(f"Total configs explored                   : {n_total}")
    print(f"Baseline (Optuna 30 trials, sequential)  : {BASELINE}")
    print(f"Hyperband 3-stage ({N_BATCHES}×{N_BATCH}→{TOP_S2}→{TOP_S3})          : {best['test_mae']:.4f}")
    print(f"Delta                                    : {BASELINE - best['test_mae']:+.4f} EUR/MWh")
    print(f"\nBest params (seed for next run):")
    print(json.dumps(best["params"], indent=2))
    print(f"\nMLflow UI: run `mlflow ui --port 5001` then open http://localhost:5001")

    # Save to agent run history so the LLM agent can learn from past runs
    try:
        from datetime import datetime
        from agent.history import save as save_history
        save_history({
            "timestamp":     datetime.now().isoformat(),
            "model_type":    MODEL_TYPE,
            "n_trials":      n_total,
            "reasoning":     "CLI hyperband run",
            "search_space":  "orchestrator default",
            "best_test_mae": best["test_mae"],
            "best_val_mae":  best["val_mae"],
            "best_params":   best["params"],
        })
    except Exception:
        pass  # history logging is optional


if __name__ == "__main__":
    run_hyperband()
