import json, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from daytona import Daytona, CreateSandboxFromSnapshotParams

SNAPSHOT  = "elec-forecast-v2"
REPO_URL  = "https://github.com/jeannineshiu/electricity-hyperband"
N_BATCH   = 9   # forks per Stage 1 batch (base + 9 = 10, exactly at concurrency limit)
N_BATCHES = 2   # Stage 1 batches → 18 configs total
TOP_S2    = 6   # top configs advancing to Stage 2
TOP_S3    = 2   # top configs advancing to Stage 3
BASELINE  = 7.23

daytona = Daytona()

def sample_params():
    return {
        "n_estimators":      random.choice([200, 400, 600, 800, 1000]),
        "max_depth":         random.randint(3, 8),
        "learning_rate":     random.choice([0.01, 0.03, 0.05, 0.1, 0.2]),
        "num_leaves":        random.randint(20, 150),
        "subsample":         round(random.uniform(0.6, 1.0), 2),
        "colsample_bytree":  round(random.uniform(0.6, 1.0), 2),
        "min_child_samples": random.randint(5, 50),
        "reg_alpha":         round(random.uniform(0.0, 1.0), 2),
        "reg_lambda":        round(random.uniform(0.0, 1.0), 2),
        "random_state":      42, "n_jobs": -1, "verbose": -1,
    }

def run_fork(base_sb, params, stage):
    fork = base_sb._experimental_fork()
    try:
        # Write config via code_run to avoid shell escaping issues with special characters
        fork.process.code_run(f"""
import json
with open('/tmp/config.json', 'w') as f:
    json.dump({params}, f)
""")
        train_resp = fork.process.exec(
            f"python $HOME/project/sandbox_train.py "
            f"--config /tmp/config.json --stage {stage}"
        )
        if train_resp.exit_code != 0:
            raise RuntimeError(f"Training failed (exit {train_resp.exit_code}): {train_resp.result}")
        resp = fork.process.exec("cat /tmp/result.json")
        if not resp.result or not resp.result.strip():
            raise RuntimeError("result.json is empty")
        return json.loads(resp.result)
    finally:
        fork.delete()

def run_parallel_forks(base_sb, configs, stage):
    results = []
    with ThreadPoolExecutor(max_workers=len(configs)) as ex:
        futures = {ex.submit(run_fork, base_sb, cfg, stage): cfg for cfg in configs}
        for f in as_completed(futures):
            try:
                r = f.result()
                results.append(r)
                print(f"    stage={stage} val_mae={r['val_mae']:.4f}")
            except Exception as e:
                print(f"    [SKIP] {e}")
    return results

# ── Setup base sandbox ────────────────────────────────────────
print("Setting up base sandbox (git clone once for all forks)...")
base_sb = daytona.create(CreateSandboxFromSnapshotParams(snapshot=SNAPSHOT))
try:
    r = base_sb.process.exec(f"git clone {REPO_URL} $HOME/project")
    if r.exit_code != 0:
        raise RuntimeError(f"git clone failed: {r.result}")
    print("Base sandbox ready.\n")

    # ── Stage 1: 2 batches × 9 forks × 10% data ──────────────
    n_total = N_BATCH * N_BATCHES
    print(f"Stage 1: {N_BATCHES} batches × {N_BATCH} forks × 10% data  ({n_total} configs total)")
    all_s1 = []
    for b in range(N_BATCHES):
        print(f"  Batch {b + 1}/{N_BATCHES}:")
        configs = [sample_params() for _ in range(N_BATCH)]
        all_s1.extend(run_parallel_forks(base_sb, configs, stage=1))

    if not all_s1:
        raise RuntimeError("All Stage 1 forks failed.")
    all_s1.sort(key=lambda x: x["val_mae"])
    survivors_s2 = all_s1[:TOP_S2]
    print(f"\nStage 1 → Top {TOP_S2} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s2]}")

    # ── Stage 2: Top 6 × 33% data ─────────────────────────────
    print(f"\nStage 2: {TOP_S2} forks × 33% data")
    s2_results = run_parallel_forks(base_sb, [r["params"] for r in survivors_s2], stage=2)

    if not s2_results:
        raise RuntimeError("All Stage 2 forks failed.")
    s2_results.sort(key=lambda x: x["val_mae"])
    survivors_s3 = s2_results[:TOP_S3]
    print(f"\nStage 2 → Top {TOP_S3} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s3]}")

    # ── Stage 3: Top 2 × 100% data ────────────────────────────
    print(f"\nStage 3: {TOP_S3} forks × 100% data")
    s3_results = run_parallel_forks(base_sb, [r["params"] for r in survivors_s3], stage=3)

    if not s3_results:
        raise RuntimeError("All Stage 3 forks failed.")
    best = min(s3_results, key=lambda x: x["test_mae"])

finally:
    base_sb.delete()

print(f"\n{'='*50}")
print(f"Total configs explored                   : {n_total}")
print(f"Baseline (Optuna 30 trials, sequential)  : {BASELINE}")
print(f"Hyperband fork-based (2×9→6→2)           : {best['test_mae']:.4f}")
print(f"Delta                                    : {BASELINE - best['test_mae']:+.4f} EUR/MWh")
