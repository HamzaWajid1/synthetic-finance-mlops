# scripts/run_pipeline_mlflow.py
import pathlib
import sys
import os

# ==============================
# 1. Ensure project root is in Python path
# ==============================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==============================
# 2. Import your modules
# ==============================
from src.train import run_anomaly_supervised_pipeline
from src.evaluate import evaluate_models
import mlflow
import mlflow.sklearn

# ==============================
# 3. Define pipeline
# ==============================
def run_full_pipeline(data_folder: str):
    # Step 1: Train models
    print("🔹 Running anomaly + supervised pipeline...")
    models, X_test, y_test = run_anomaly_supervised_pipeline(data_folder)

    # Step 2: Evaluate models
    print("🔹 Evaluating models...")
    results_df = evaluate_models(models, X_test, y_test)
    print("\nEvaluation results:\n", results_df)

    # Step 3: MLflow logging
    print("🔹 Logging results to MLflow...")
    mlflow.set_experiment("finance_anomaly_supervised_pipeline")

    with mlflow.start_run():
        # Log model parameters
        for name, model in models.items():
            if hasattr(model, "get_params"):
                params = model.get_params()
                for key, value in params.items():
                    mlflow.log_param(f"{name}_{key}", value)

        # Log evaluation metrics
        for name in results_df.index:
            for metric in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "Accuracy", "Balanced Accuracy"]:
                mlflow.log_metric(f"{name}_{metric}", results_df.loc[name, metric])

        # Path to store MLflow artifacts locally under 'models/'
        artifact_base_path = pathlib.Path("models").absolute()
        artifact_base_path.mkdir(exist_ok=True, parents=True)

        # Log models properly (no slashes in name)
        for name, model in models.items():
            model_name = name.replace(" ", "_")  # e.g., Logistic_Regression
            mlflow.sklearn.log_model(model, artifact_path=model_name)

        print("✅ MLflow logging complete!")

# ==============================
# 4. Entry point
# ==============================
if __name__ == "__main__":
    data_folder = "data/raw"  # replace with your actual path
    run_full_pipeline(data_folder)

