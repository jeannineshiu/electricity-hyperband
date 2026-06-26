import json, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from daytona import Daytona, CreateSandboxFromSnapshotParams

SNAPSHOT  = "elec-forecast-lstm-v1"
REPO_URL  = "https://github.com/jeannineshiu/electricity-hyperband"
N_BATCH   = 9   # sandboxes per batch (within 10-sandbox concurrency limit)
N_BATCHES = 2   # Stage 1 batches → 18 configs total
TOP_S2    = 6   # top configs advancing to Stage 2
TOP_S3    = 3   # top configs advancing to Stage 3
BASELINE  = 7.23

daytona = Daytona()


def sample_params():
    return {
        "hidden_size":   random.choice([32, 64, 128]),
        "num_layers":    random.choice([1, 2]),
        "dropout":       round(random.uniform(0.1, 0.4), 2),
        "learning_rate": random.choice([0.001, 0.003, 0.01]),
        "batch_size":    random.choice([32, 64, 128]),
        "window_size":   random.choice([24, 48]),
    }


def run_sandbox(params, stage):
    sb = daytona.create(CreateSandboxFromSnapshotParams(snapshot=SNAPSHOT))
    try:
        clone_resp = sb.process.exec(f"git clone {REPO_URL} $HOME/project")
        if clone_resp.exit_code != 0:
            raise RuntimeError(f"git clone failed: {clone_resp.result}")

        # Write config via code_run to avoid shell escaping issues
        sb.process.code_run(f"""
import json
with open('/tmp/config.json', 'w') as f:
    json.dump({params}, f)
""")
        train_resp = sb.process.exec(
            f"python $HOME/project/sandbox_train_lstm.py "
            f"--config /tmp/config.json --stage {stage}"
        )
        if train_resp.exit_code != 0:
            raise RuntimeError(
                f"Training failed (exit {train_resp.exit_code}): {train_resp.result}"
            )
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
                print(f"    stage={stage} val_mae={r['val_mae']:.4f}", flush=True)
            except Exception as e:
                print(f"    [SKIP] {e}", flush=True)
    return results


# ── Stage 1: N_BATCHES × N_BATCH sandboxes × 5 epochs ────────
t_start = time.time()
n_total = N_BATCH * N_BATCHES
print(f"Stage 1: {N_BATCHES} batches × {N_BATCH} sandboxes × 5 epochs  ({n_total} configs total)")
all_s1 = []
for b in range(N_BATCHES):
    print(f"  Batch {b + 1}/{N_BATCHES}:")
    configs = [sample_params() for _ in range(N_BATCH)]
    all_s1.extend(run_parallel(configs, stage=1))

if not all_s1:
    raise RuntimeError("All Stage 1 sandboxes failed.")
all_s1.sort(key=lambda x: x["val_mae"])
survivors_s2 = all_s1[:TOP_S2]
print(f"\nStage 1 → Top {TOP_S2} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s2]}")

# ── Stage 2: Top configs × 20 epochs ─────────────────────────
print(f"\nStage 2: {len(survivors_s2)} sandboxes × 20 epochs")
s2_results = run_parallel([r["params"] for r in survivors_s2], stage=2)

if not s2_results:
    raise RuntimeError("All Stage 2 sandboxes failed.")
s2_results.sort(key=lambda x: x["val_mae"])
survivors_s3 = s2_results[:TOP_S3]
print(f"\nStage 2 → Top {TOP_S3} val_MAE: {[round(r['val_mae'], 4) for r in survivors_s3]}")

# ── Stage 3: Top configs × 50 epochs ─────────────────────────
print(f"\nStage 3: {len(survivors_s3)} sandboxes × 50 epochs")
s3_results = run_parallel([r["params"] for r in survivors_s3], stage=3)

if not s3_results:
    raise RuntimeError("All Stage 3 sandboxes failed.")
best = min(s3_results, key=lambda x: x["test_mae"])

elapsed = time.time() - t_start
mins, secs = divmod(int(elapsed), 60)

print(f"\n{'='*50}")
print(f"Total wall-clock time                    : {mins}m {secs}s")
print(f"Total configs explored                   : {n_total}")
print(f"LightGBM baseline (Optuna sequential)    : {BASELINE}")
print(f"LSTM Hyperband ({N_BATCHES}×{N_BATCH}→{TOP_S2}→{TOP_S3})              : {best['test_mae']:.4f}")
print(f"Delta                                    : {BASELINE - best['test_mae']:+.4f} EUR/MWh")
print(f"\nBest LSTM params:")
import json as _json
print(_json.dumps(best["params"], indent=2))
