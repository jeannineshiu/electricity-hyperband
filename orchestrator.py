import json, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from daytona import Daytona, CreateSandboxFromSnapshotParams

SNAPSHOT  = "elec-forecast-v1"
N_STAGE1  = 20
TOP_K     = 6
BASELINE  = 7.23

daytona = Daytona()

def sample_params():
    return {
        "n_estimators":      random.choice([200, 400, 600, 800]),
        "max_depth":         random.randint(3, 8),
        "learning_rate":     random.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
        "num_leaves":        random.randint(20, 150),
        "subsample":         round(random.uniform(0.6, 1.0), 2),
        "colsample_bytree":  round(random.uniform(0.6, 1.0), 2),
        "min_child_samples": random.randint(5, 50),
        "random_state":      42, "n_jobs": -1, "verbose": -1,
    }

def run_sandbox(params, stage):
    sb = daytona.create(CreateSandboxFromSnapshotParams(snapshot=SNAPSHOT))
    try:
        # 用 code_run 寫 config（比 echo 安全，不會被特殊字元爆掉）
        sb.process.code_run(f"""
import json
with open('/tmp/config.json', 'w') as f:
    json.dump({params}, f)
""")
        sb.process.exec(
            "python /workspace/project/sandbox_train.py "
            f"--config /tmp/config.json --stage {stage}"
        )
        resp = sb.process.exec("cat /tmp/result.json")
        return json.loads(resp.result)
    finally:
        sb.delete()

# ── Stage 1 ────────────────────────────────────────────────
print(f"Stage 1: {N_STAGE1} sandboxes × 15% data")
configs = [sample_params() for _ in range(N_STAGE1)]
s1_results = []

with ThreadPoolExecutor(max_workers=N_STAGE1) as ex:
    futures = {ex.submit(run_sandbox, cfg, 1): cfg for cfg in configs}
    for f in as_completed(futures):
        r = f.result()
        s1_results.append(r)
        print(f"  val_mae={r['val_mae']:.4f}")

s1_results.sort(key=lambda x: x["val_mae"])
survivors = s1_results[:TOP_K]
print(f"\nSurvivors val_MAE: {[round(r['val_mae'],4) for r in survivors]}")

# ── Stage 2 ────────────────────────────────────────────────
print(f"\nStage 2: {TOP_K} sandboxes × full data")
s2_results = []

with ThreadPoolExecutor(max_workers=TOP_K) as ex:
    futures = [ex.submit(run_sandbox, r["params"], 2) for r in survivors]
    for f in as_completed(futures):
        r = f.result()
        s2_results.append(r)
        print(f"  test_mae={r['test_mae']:.4f}")

best = min(s2_results, key=lambda x: x["test_mae"])
print(f"\n{'='*45}")
print(f"Baseline (Optuna 30 trials, sequential) : {BASELINE}")
print(f"Hyperband search (20→6, parallel)       : {best['test_mae']:.4f}")
print(f"Delta                                   : {BASELINE - best['test_mae']:+.4f} EUR/MWh")
