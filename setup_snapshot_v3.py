from daytona import Daytona

SNAPSHOT_NAME = "elec-forecast-v3"

daytona = Daytona()
sb = daytona.create()

r = sb.process.exec(
    "pip install lightgbm xgboost catboost scikit-learn pandas pyarrow numpy -q"
)
if r.exit_code != 0:
    sb.delete()
    raise RuntimeError(f"pip install failed: {r.result}")
print("pip install OK")

r = sb.process.exec(
    "python -c \""
    "import lightgbm, xgboost, catboost, sklearn, pandas; "
    "print('lgb', lightgbm.__version__, '| xgb', xgboost.__version__, "
    "'| cat', catboost.__version__, '| sklearn', sklearn.__version__)"
    "\""
)
if r.exit_code != 0:
    sb.delete()
    raise RuntimeError(f"Package verification failed: {r.result}")
print(r.result.strip())

sb._experimental_create_snapshot(SNAPSHOT_NAME)
sb.delete()
print(f"Snapshot '{SNAPSHOT_NAME}' ready.")
