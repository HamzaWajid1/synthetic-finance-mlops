# scripts/run_pipeline_mlflow.py
import pathlib
import sys
import os
from joblib import dump
from src.train import run_anomaly_supervised_pipeline
from src.evaluate import evaluate_models
import mlflow
import mlflow.sklearn

def run_full_pipeline(data_folder: str, save_local: bool = False):
    """
    Runs anomaly + supervised pipeline, evaluates models, logs to MLflow,
    and optionally saves local copies.

    Parameters
    ----------
    data_folder : str
        Path to raw data folder
    save_local : bool
        If True, saves models locally under project_root/models/
    """

    # ------------------------------
    # 1. Train models
    # ------------------------------
    print("🔹 Running anomaly + supervised pipeline...")
    models, X_test, y_test = run_anomaly_supervised_pipeline(data_folder)

    # ------------------------------
    # 2. Evaluate models
    # ------------------------------
    print("🔹 Evaluating models...")
    results_df = evaluate_models(models, X_test, y_test)
    print("\nEvaluation results:\n", results_df)

    # ------------------------------
    # 3. MLflow logging
    # ------------------------------
    print("🔹 Logging results to MLflow...")
    mlflow.set_experiment("finance_anomaly_supervised_pipeline")

    for name, model in models.items():
        model_name = name.replace(" ", "_")

        with mlflow.start_run(run_name=model_name):

            # Log parameters
            if hasattr(model, "get_params"):
                for key, value in model.get_params().items():
                    mlflow.log_param(key, value)

            # Log metrics
            for metric in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "Accuracy", "Balanced Accuracy"]:
                mlflow.log_metric(metric, results_df.loc[name, metric])

            # Log model in MLflow
            mlflow.sklearn.log_model(model, artifact_path="models",registered_model_name=model_name)

            print(f"✅ Logged model '{model_name}' to MLflow.")
                    
            # Save local copy
            if save_local:
                local_dir = pathlib.Path("models").absolute()
                local_dir.mkdir(exist_ok=True, parents=True)
                dump(model, local_dir / f"{model_name}.joblib")
                print(f"💾 Saved local copy at '{local_dir / f'{model_name}.joblib'}'")

    print("🎉 Pipeline complete! All models logged and saved.")

# ------------------------------
# 4. Entry point
# ------------------------------
if __name__ == "__main__":
    data_folder = "data/raw"  # replace with your actual path
    run_full_pipeline(data_folder)


