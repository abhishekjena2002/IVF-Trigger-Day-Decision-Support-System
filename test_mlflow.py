
import mlflow

mlflow.set_experiment("MLFLOW_TEST")

with mlflow.start_run():
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("test_metric", 0.99)

print("MLflow test run completed")
