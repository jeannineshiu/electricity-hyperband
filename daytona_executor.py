"""
Daytona sandbox management.

Responsible for:
  - Creating a sandbox from the project snapshot
  - Cloning the repo, writing the config, running sandbox_train.py
  - Collecting the result and deleting the sandbox
"""

import json
from daytona import Daytona, CreateSandboxFromSnapshotParams
import config

daytona = Daytona()


def run_sandbox(params: dict, stage: int) -> dict:
    """
    Creates one Daytona sandbox, runs training, returns result dict.
    Reads config.SNAPSHOT, config.REPO_URL, config.MODEL_TYPE at call time.
    """
    sb = daytona.create(CreateSandboxFromSnapshotParams(snapshot=config.SNAPSHOT))
    try:
        clone_resp = sb.process.exec(f"git clone {config.REPO_URL} $HOME/project")
        if clone_resp.exit_code != 0:
            raise RuntimeError(f"git clone failed: {clone_resp.result}")

        # Write config via code_run to avoid shell escaping issues
        sb.process.code_run(f"""
import json
with open('/tmp/config.json', 'w') as f:
    json.dump({params}, f)
""")
        train_resp = sb.process.exec(
            f"python $HOME/project/sandbox_train.py "
            f"--config /tmp/config.json --stage {stage} --model {config.MODEL_TYPE}"
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
