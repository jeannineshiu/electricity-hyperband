import mlflow

_trial_counter = 0


def start_experiment(experiment_name: str = "electricity-hyperband"):
    global _trial_counter
    _trial_counter = 0
    mlflow.set_experiment(experiment_name)


def log_trial(params: dict, val_mae: float, test_mae: float, stage: int, batch: int = None):
    tags = {"stage": str(stage)}
    if batch is not None:
        tags["batch"] = str(batch)

    global _trial_counter
    _trial_counter += 1
    run_name = f"s{stage}_trial_{_trial_counter:03d}"
    if batch is not None:
        run_name = f"s{stage}_b{batch}_trial_{_trial_counter:03d}"

    with mlflow.start_run(run_name=run_name, nested=True, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("test_mae", test_mae)


def log_hyperband_summary(best_params: dict, best_test_mae: float, baseline: float, n_total: int, elapsed_sec: int):
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_test_mae", best_test_mae)
    mlflow.log_metric("baseline_test_mae", baseline)
    mlflow.log_metric("improvement_eur_mwh", baseline - best_test_mae)
    mlflow.log_metric("configs_explored", n_total)
    mlflow.log_metric("wall_clock_sec", elapsed_sec)
