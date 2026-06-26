from daytona import Daytona

SNAPSHOT_NAME = "elec-forecast-v2"

daytona = Daytona()
sb = daytona.create()

r = sb.process.exec("pip install lightgbm pandas scikit-learn pyarrow numpy -q")
if r.exit_code != 0:
    sb.delete()
    raise RuntimeError(f"pip install failed (exit {r.exit_code}): {r.result}")
print("pip install OK")

# Verify packages are importable
r = sb.process.exec("python -c \"import lightgbm, pandas, sklearn, pyarrow; print('packages OK')\"")
if r.exit_code != 0:
    sb.delete()
    raise RuntimeError(f"Package verification failed: {r.result}")
print(r.result.strip())

sb._experimental_create_snapshot(SNAPSHOT_NAME)
sb.delete()
print(f"Snapshot '{SNAPSHOT_NAME}' ready.")
