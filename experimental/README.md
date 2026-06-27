# Experimental Features

This directory contains features that are **architecturally complete but not yet verified end-to-end**.

---

## LSTM Hyperband

Files:
- `orchestrator_lstm.py` — 3-stage Hyperband orchestrator for LSTM
- `sandbox_train_lstm.py` — PyTorch LSTM training script (runs inside Daytona sandbox)
- `setup_snapshot_lstm.py` — Creates `elec-forecast-lstm-v1` snapshot with PyTorch installed

### Architecture

The LSTM uses the same 3-stage Hyperband structure as the main LightGBM pipeline, with stage-aware epoch limits instead of data fractions:

| Stage | Max Epochs | Early Stopping |
|---|---|---|
| Stage 1 | 5 | patience=2 |
| Stage 2 | 20 | patience=5 |
| Stage 3 | 50 | patience=10 |

Hyperparameter search space: `hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size`, `window_size`

The model uses GPU automatically if available (`torch.cuda.is_available()`).

### Known Issue

The orchestrator hangs during execution without producing output. Suspected cause:

- PyTorch training on Daytona CPU VMs is significantly slower than estimated
- Stage 1 with 5 epochs may take 3–10 minutes per sandbox (vs seconds for LightGBM)
- Without `flush=True` on all print statements, buffered output doesn't appear until completion

### To Resume Development

1. Add per-epoch progress prints inside `sandbox_train_lstm.py`
2. Reduce Stage 1 to 2–3 epochs as a first sanity check
3. Run with `N_BATCHES=1, N_BATCH=1` (single sandbox) to isolate timing
4. Consider capping `window_size` to 24 (not 48) to reduce sequence length

### Why It's Experimental

The main pipeline (LightGBM + generic model interface) already demonstrates the Daytona parallelism story. LSTM adds architectural depth but requires additional debugging time to verify.
