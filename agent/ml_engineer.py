"""
ML Engineer Agent — powered by Claude API + Daytona sandboxes.

Usage:
    python agent/ml_engineer.py "My model is overfitting on validation"
    python agent/ml_engineer.py  # uses default prompt
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orchestrator as orch
from orchestrator import MODEL_DEFAULTS as _DEFAULTS, MODEL_BOUNDS as _BOUNDS
from agent.history import load as load_history, save as save_history

MODEL_ID = "claude-opus-4-8"
BASELINE  = 7.23

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an ML Engineer agent specializing in electricity price forecasting.

Project context:
- Task: Predict hourly day-ahead electricity prices (EUR/MWh) using ENTSO-E market data
- Data split: Train 2020-2022, Validation 2023, Test 2024 (43,680 hourly rows)
- Baseline (Optuna 30 trials, sequential): 7.23 EUR/MWh test MAE
- Best found so far (Hyperband parallel): 7.1754 EUR/MWh test MAE (LightGBM)
- Key challenge: 2023→2024 distribution shift causes a val/test MAE gap

Available models: lightgbm, xgboost, catboost, rf

Your workflow:
1. ALWAYS call read_experiment_history first to understand what has been tried
2. Analyze the problem described by the user
3. Propose a targeted search space that addresses the issue
4. Launch Hyperband with that refined space
5. Report findings: what changed, what MAE was achieved, what to try next

Search space guidance:
- Overfitting (val << test): increase reg_lambda/reg_alpha, reduce num_leaves/depth, increase min_child_samples
- Underfitting (both high): increase capacity, lower regularization
- Val/test gap: prefer conservative models with higher regularization
- Search space format: {"param": {"type": "choice", "values": [v1, v2]}} or {"param": {"type": "range", "min": X, "max": Y}}"""

# ── Tool definitions ──────────────────────────────────────────
TOOLS = [
    {
        "name": "read_experiment_history",
        "description": (
            "Read the history of past Hyperband runs — best configs, val_mae, test_mae, "
            "model types, and the reasoning behind each search. "
            "Always call this first before suggesting a new search space."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_recent": {
                    "type": "integer",
                    "description": "Number of most recent runs to return (default: 5)"
                }
            },
            "required": []
        }
    },
    {
        "name": "launch_hyperband",
        "description": (
            "Launch a targeted 3-stage Hyperband search using Daytona parallel sandboxes. "
            "Stage 1 (10% data) → Stage 2 (33%) → Stage 3 (100%). "
            "Returns best test MAE and configuration found. Saves result to run history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": ["lightgbm", "xgboost", "catboost", "rf"],
                    "description": "Model type to search"
                },
                "search_space": {
                    "type": "object",
                    "description": (
                        "Refined hyperparameter search space. "
                        "Format: {\"param\": {\"type\": \"choice\", \"values\": [v1, v2, ...]}} "
                        "or {\"param\": {\"type\": \"range\", \"min\": X, \"max\": Y}}"
                    )
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Stage 1 configs to try in parallel (default: 9, max: 9)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you chose this search space — logged to run history"
                }
            },
            "required": ["model_type", "search_space", "reasoning"]
        }
    }
]

# _BOUNDS and _DEFAULTS are imported from orchestrator — single source of truth


def _validate_search_space(model_type: str, search_space: dict) -> list[str]:
    errors = []
    bounds = _BOUNDS.get(model_type, {})
    for param, spec in search_space.items():
        if param not in bounds:
            continue
        lo, hi = bounds[param]
        values = spec.get("values") or [spec.get("min"), spec.get("max")]
        for v in (v for v in values if v is not None):
            if not (lo <= v <= hi):
                errors.append(f"{param}={v} out of valid range [{lo}, {hi}]")
    return errors


def _make_sampler(model_type: str, search_space: dict):
    defaults = dict(_DEFAULTS.get(model_type, {}))

    def sample():
        params = dict(defaults)
        for key, spec in search_space.items():
            if spec["type"] == "choice":
                params[key] = random.choice(spec["values"])
            elif spec["type"] == "range":
                lo, hi = spec["min"], spec["max"]
                if isinstance(lo, float) or isinstance(hi, float):
                    params[key] = round(random.uniform(lo, hi), 4)
                else:
                    params[key] = random.randint(int(lo), int(hi))
        return params

    return sample


def _run_mini_hyperband(model_type: str, sampler, n_trials: int) -> dict:
    """3-stage mini-Hyperband: n_trials → top 3 → best 1."""
    orch.MODEL_TYPE = model_type

    print(f"\n  [Stage 1] {n_trials} configs × 10% data", flush=True)
    s1 = [r for r in orch.stream_stage([sampler() for _ in range(n_trials)], stage=1)]
    if not s1:
        return {"error": "All Stage 1 sandboxes failed"}
    s1.sort(key=lambda x: x["val_mae"])
    top3 = s1[:min(3, len(s1))]
    print(f"  [Stage 1] top-3 val_mae: {[round(r['val_mae'], 4) for r in top3]}", flush=True)

    print(f"\n  [Stage 2] {len(top3)} configs × 33% data", flush=True)
    s2 = [r for r in orch.stream_stage([r["params"] for r in top3], stage=2)]
    if not s2:
        return {"error": "All Stage 2 sandboxes failed"}
    s2.sort(key=lambda x: x["val_mae"])
    best_s2 = s2[0]
    print(f"  [Stage 2] best val_mae: {best_s2['val_mae']:.4f}", flush=True)

    print(f"\n  [Stage 3] 1 config × 100% data", flush=True)
    s3 = [r for r in orch.stream_stage([best_s2["params"]], stage=3)]
    if not s3:
        return {"error": "Stage 3 failed"}

    return s3[0]


# ── Tool executors ────────────────────────────────────────────
def _exec_read_history(n_recent: int = 5) -> str:
    history = load_history(n_recent)
    if not history:
        return json.dumps({
            "message": "No experiment history yet. This is the first run.",
            "runs": []
        })
    return json.dumps({
        "total_runs_on_file": len(load_history()),
        "showing": len(history),
        "baseline_test_mae": BASELINE,
        "runs": [
            {
                "timestamp":    r.get("timestamp"),
                "model_type":   r.get("model_type"),
                "n_trials":     r.get("n_trials"),
                "best_test_mae": r.get("best_test_mae"),
                "best_val_mae": r.get("best_val_mae"),
                "reasoning":    r.get("reasoning", "—"),
                "best_params":  r.get("best_params"),
            }
            for r in history
        ]
    }, indent=2)


def _exec_launch_hyperband(
    model_type: str,
    search_space: dict,
    n_trials: int = 9,
    reasoning: str = "",
) -> str:
    errors = _validate_search_space(model_type, search_space)
    if errors:
        return json.dumps({"error": f"Invalid search space: {errors}"})

    n_trials = max(1, min(n_trials, 9))
    sampler  = _make_sampler(model_type, search_space)
    result   = _run_mini_hyperband(model_type, sampler, n_trials)

    if "error" in result:
        return json.dumps(result)

    save_history({
        "timestamp":    datetime.now().isoformat(),
        "model_type":   model_type,
        "n_trials":     n_trials,
        "reasoning":    reasoning,
        "search_space": search_space,
        "best_test_mae": result["test_mae"],
        "best_val_mae":  result["val_mae"],
        "best_params":   result["params"],
    })

    return json.dumps({
        "best_test_mae": result["test_mae"],
        "best_val_mae":  result["val_mae"],
        "delta_vs_baseline": round(BASELINE - result["test_mae"], 4),
        "best_params":   result["params"],
        "n_trials_run":  n_trials,
    }, indent=2)


def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "read_experiment_history":
        return _exec_read_history(tool_input.get("n_recent", 5))
    if name == "launch_hyperband":
        return _exec_launch_hyperband(
            model_type  = tool_input["model_type"],
            search_space= tool_input["search_space"],
            n_trials    = tool_input.get("n_trials", 9),
            reasoning   = tool_input.get("reasoning", ""),
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Main agent loop ───────────────────────────────────────────
def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next(
                (b.text for b in response.content if hasattr(b, "text")),
                "(No text response)"
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\nAgent → {block.name}", flush=True)
                    result = _execute_tool(block.name, block.input)
                    preview = result[:300] + ("..." if len(result) > 300 else "")
                    print(f"  ↳ {preview}", flush=True)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        return f"Unexpected stop reason: {response.stop_reason}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Engineer Agent")
    parser.add_argument(
        "message", nargs="?",
        default="Analyze my experiment history and suggest what to try next to beat the 7.23 baseline.",
    )
    args = parser.parse_args()

    print(f"User: {args.message}\n")
    answer = run_agent(args.message)
    print(f"\nAgent: {answer}\n")
