import sys, os, time
from datetime import datetime
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrator as orch
from orchestrator import (
    stream_stage, get_seed_params,
    N_BATCH, N_BATCHES, TOP_S2, TOP_S3, BASELINE,
)
from agent.history import save as save_history

st.set_page_config(
    page_title="Hyperband Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Electricity Price — Hyperband Dashboard")
st.caption("Parallel hyperparameter search with Daytona sandboxes · LightGBM · ENTSO-E 2020–2024")

# ── Pipeline overview ──────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Stage 1", f"{N_BATCH * N_BATCHES} configs", f"{N_BATCH} sandboxes / batch · 10% data")
c2.metric("Stage 2", f"Top {TOP_S2} + seeds", "33% data")
c3.metric("Stage 3", f"Top {TOP_S3}", "100% data · full training")

# ── Model selector ─────────────────────────────────────────────
model_choice = st.selectbox(
    "Model",
    options=["lightgbm", "xgboost", "catboost", "rf"],
    format_func=lambda x: {
        "lightgbm": "LightGBM",
        "xgboost":  "XGBoost",
        "catboost": "CatBoost",
        "rf":       "Random Forest",
    }[x],
)
orch.MODEL_TYPE = model_choice  # update module-level variable

st.divider()

if not st.button("▶ Start Hyperband Search", type="primary"):
    st.info("Click **Start** to launch the parallel Hyperband search across Daytona sandboxes.")
    st.stop()

# ── Search starts ──────────────────────────────────────────────
n_total  = N_BATCH * N_BATCHES
t_start  = time.time()
all_rows: list[dict] = []

stage_label  = st.empty()
progress_bar = st.progress(0.0)
progress_txt = st.empty()
leaderboard  = st.empty()
best_box     = st.empty()


def refresh(completed: int, stage: int):
    progress_bar.progress(min(completed / n_total, 1.0))
    progress_txt.caption(f"Stage {stage} · {completed} / {n_total} configs complete")
    if not all_rows:
        return
    df = (
        pd.DataFrame(all_rows)
        .sort_values("val_mae")
        .reset_index(drop=True)
    )
    df.index += 1
    df["batch"] = df["batch"].fillna("—").astype(str).replace("nan", "—")
    leaderboard.dataframe(
        df[["stage", "batch", "val_mae", "test_mae"]].round(4),
        use_container_width=True,
        height=min(35 * (len(df) + 1) + 10, 420),
    )
    best_val = df["val_mae"].min()
    best_box.metric(
        "Best val_mae so far",
        f"{best_val:.4f}",
        delta=f"{BASELINE - best_val:+.4f} vs baseline {BASELINE}",
        delta_color="normal",
    )


# ── Stage 1 ────────────────────────────────────────────────────
stage_label.info(f"🔍 **Stage 1** — Fast Screen · 10% data · {N_BATCH} sandboxes per batch")
all_s1 = []
completed = 0

for b in range(N_BATCHES):
    configs = [orch.sample_params() for _ in range(N_BATCH)]
    for r in stream_stage(configs, stage=1):
        all_s1.append(r)
        all_rows.append({**r, "stage": 1, "batch": b + 1})
        completed += 1
        refresh(completed, stage=1)

if not all_s1:
    st.error("All Stage 1 sandboxes failed. Check your DAYTONA_API_KEY and disk limits.")
    st.stop()

all_s1.sort(key=lambda x: x["val_mae"])
survivors_s2 = all_s1[:TOP_S2]
for seed in get_seed_params():
    survivors_s2.append({"val_mae": 0.0, "test_mae": 0.0, "params": seed})

# ── Stage 2 ────────────────────────────────────────────────────
stage_label.info(f"⚖️ **Stage 2** — Medium Eval · 33% data · {len(survivors_s2)} sandboxes")
s2_results = []

for r in stream_stage([x["params"] for x in survivors_s2], stage=2):
    s2_results.append(r)
    all_rows.append({**r, "stage": 2, "batch": None})
    refresh(completed, stage=2)

if not s2_results:
    st.error("All Stage 2 sandboxes failed.")
    st.stop()

s2_results.sort(key=lambda x: x["val_mae"])
survivors_s3 = s2_results[:TOP_S3]

# ── Stage 3 ────────────────────────────────────────────────────
stage_label.info(f"🏆 **Stage 3** — Full Training · 100% data · {len(survivors_s3)} sandboxes")
s3_results = []

for r in stream_stage([x["params"] for x in survivors_s3], stage=3):
    s3_results.append(r)
    all_rows.append({**r, "stage": 3, "batch": None})
    refresh(completed, stage=3)

if not s3_results:
    st.error("All Stage 3 sandboxes failed.")
    st.stop()

# ── Final results ───────────────────────────────────────────────
best    = min(s3_results, key=lambda x: x["test_mae"])
elapsed = int(time.time() - t_start)

save_history({
    "timestamp":     datetime.now().isoformat(),
    "model_type":    orch.MODEL_TYPE,
    "n_trials":      n_total,
    "reasoning":     "dashboard run",
    "search_space":  "orchestrator default",
    "best_test_mae": best["test_mae"],
    "best_val_mae":  best["val_mae"],
    "best_params":   best["params"],
})

stage_label.success(f"✅ Search complete in {elapsed}s · {n_total} configs explored")
progress_bar.progress(1.0)
progress_txt.empty()

st.divider()
st.subheader("Final Results")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Test MAE",      f"{best['test_mae']:.4f}",
            delta=f"{BASELINE - best['test_mae']:+.4f} EUR/MWh")
col2.metric("Baseline (Optuna)",  f"{BASELINE}")
col3.metric("Configs Explored",   n_total)
col4.metric("Wall-clock Time",    f"{elapsed}s")

st.subheader("Best Hyperparameters Found")
st.json(best["params"])
