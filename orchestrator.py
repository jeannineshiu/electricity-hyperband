import json, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from daytona import Daytona, CreateSandboxFromSnapshotParams

SNAPSHOT  = "elec-forecast-v2"
REPO_URL  = "https://github.com/jeannineshiu/electricity-hyperband"
N_BATCH   = 9   # sandboxes per batch (within the 10-sandbox concurrency limit)
N_BATCHES = 4   # Stage 1 batches → 36 configs total
TOP_S2    = 6   # top configs advancing to Stage 2
TOP_S3    = 3   # top configs advancing to Stage 3
BASELINE  = 7.23

# Known good configs from previous runs — always included in Stage 2
# Paste best params output here after each successful run
SEED_PARAMS = []

daytona = Daytona()

def sample_params():
    return {
        # More trees with lower LR → better generalization
        "n_estimators":      random.choice([500, 800, 1000, 1500, 2000]),
        "max_depth":         random.randint(3, 6),          # shallower trees reduce overfitting
        "learning_rate":     random.choice([0.005, 0.01, 0.03, 0.05]),
        "num_leaves":        random.randint(15, 63),        # cap at 63 (2^6-1) for depth-6 trees
        "subsample":         round(random.uniform(0.6, 0.9), 2),
        "colsample_bytree":  round(random.uniform(0.6, 0.9), 2),
        "min_child_samples": random.randint(20, 100),       # higher → more regularization
        "reg_alpha":         round(random.uniform(0.0, 2.0), 2),
        "reg_lambda":        round(random.uniform(0.1, 2.0), 2),
        "random_state":      42, "n_jobs": -1, "verbose": -1,
    }

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
            f"--config /tmp/config.json --stage {stage}"
        )
        if train_resp.exit_code != 0:
            raise RuntimeError(f"Training failed (exit {train_resp.exit_code}): {train_resp.result}")
        resp = sb.process.exec("cat /tmp/result.json")
        if not resp.result or not resp.result.strip():
            raise RuntimeError("result.json is empty")
        return json.loads(resp.result)
    finally:
        sb.delete()

def run_parallel(configs, stage):
    results = []
    with ThreadPoolExecutor(max_workers=len(configs)) as ex:
        futures = {ex.submit(run_sandbox, cfg, stage): cfg for cfg in configs}
        for f in as_completed(futures):
            try:
                r = f.result()
                results.append(r)
                print(f"    stage={stage} val_mae={r['val_mae']:.4f}")
            except Exception as e:
                print(f"    [SKIP] {e}")
    return results

# ── Stage 1: 2 batches × 9 sandboxes × 10% data ──────────────
n_total = N_BATCH * N_BATCHES
print(f"Stage 1: {N_BATCHES} batches × {N_BATCH} sandboxes × 10% data  ({n_total} configs total)")
all_s1 = []
for b in range(N_BATCHES):
    print(f"  Batch {b + 1}/{N_BATCHES}:")
    configs = [sample_params() for _ in range(N_BATCH)]
    all_s1.extend(run_parallel(configs, stage=1))

if not all_s1:
    raise RuntimeError("All Stage 1 sandboxes failed.")
all_s1.sort(key=lambda x: x["val_mae"])
survivors_s2 = all_s1[:TOP_S2]

# Inject known-good seed params directly into Stage 2
for seed in SEED_PARAMS:
    survivors_s2.append({"val_mae": 0.0, "test_mae": 0.0, "params": seed})
print(f"\nStage 1 → Top {TOP_S2} + {len(SEED_PARAMS)} seeds → Stage 2 ({len(survivors_s2)} total)")

# ── Stage 2: Top 6 × 33% data ─────────────────────────────────
print(f"\nStage 2: {len(survivors_s2)} sandboxes × 33% data")
s2_results = run_parallel([r["params"] for r in survivors_s2], stage=2)

if not s2_results:
    raise RuntimeError("All Stage 2 sandboxes failed.")
s2_results.sort(key=lambda x: x["val_mae"])
survivors_s3 = s2_results[:TOP_S3]
print(f"\nStage 2 → Top {TOP_S3} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s3]}")

# ── Stage 3: Top 2 × 100% data ────────────────────────────────
print(f"\nStage 3: {len(survivors_s3)} sandboxes × 100% data")
s3_results = run_parallel([r["params"] for r in survivors_s3], stage=3)

if not s3_results:
    raise RuntimeError("All Stage 3 sandboxes failed.")
best = min(s3_results, key=lambda x: x["test_mae"])

print(f"\n{'='*50}")
print(f"Total configs explored                   : {n_total}")
print(f"Baseline (Optuna 30 trials, sequential)  : {BASELINE}")
print(f"Hyperband 3-stage ({N_BATCHES}×{N_BATCH}→{TOP_S2}→{TOP_S3})          : {best['test_mae']:.4f}")
print(f"Delta                                    : {BASELINE - best['test_mae']:+.4f} EUR/MWh")
print(f"\nBest params (seed for next run):")
import json as _json
print(_json.dumps(best["params"], indent=2))
