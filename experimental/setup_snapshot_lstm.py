from daytona import Daytona

SNAPSHOT_NAME = "elec-forecast-lstm-v1"

daytona = Daytona()
sb = daytona.create()

# Install PyTorch (CPU-only, smaller image) + data dependencies
r = sb.process.exec(
    "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu "
    "pandas pyarrow numpy scikit-learn -q"
)
if r.exit_code != 0:
    sb.delete()
    raise RuntimeError(f"pip install failed: {r.result}")
print("pip install OK")

r = sb.process.exec(
    "python -c \"import torch, pandas, sklearn; "
    "print('torch', torch.__version__, '| pandas', pandas.__version__)\""
)
if r.exit_code != 0:
    sb.delete()
    raise RuntimeError(f"Package verification failed: {r.result}")
print(r.result.strip())

sb._experimental_create_snapshot(SNAPSHOT_NAME)
sb.delete()
print(f"Snapshot '{SNAPSHOT_NAME}' ready.")
