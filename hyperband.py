"""
Hyperband search algorithm — pure, no MLflow or history coupling.

Exposes stream_stage() as the core primitive:
  - Used by the CLI orchestrator (via run_parallel)
  - Used directly by the Streamlit dashboard
  - Used directly by the LLM agent
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from daytona_executor import run_sandbox


def stream_stage(configs: list, stage: int):
    """
    Generator: submits all configs to Daytona in parallel, yields one result
    dict per sandbox as it completes. Skips failed sandboxes silently.
    """
    with ThreadPoolExecutor(max_workers=max(1, len(configs))) as ex:
        futures = {ex.submit(run_sandbox, cfg, stage): cfg for cfg in configs}
        for f in as_completed(futures):
            try:
                yield f.result()
            except Exception as e:
                print(f"    [SKIP] {e}", flush=True)
