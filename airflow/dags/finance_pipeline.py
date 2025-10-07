from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import joblib

from src.preprocess import preprocess_data
from src.model_prep import prepare_model_data
from src.prep_data_train import run_anomaly_models
from src.evaluate import evaluate_models

# Paths
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    dag_id="synthetic_finance_pipeline",
    default_args=default_args,
    description="Synthetic Finance Anomaly Detection Pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def step1_preprocess(**kwargs):
        data_folder = "data/raw"
        df = preprocess_data(data_folder)

        preprocessed_path = os.path.join(ARTIFACTS_DIR, "df_preprocessed.parquet")
        df.to_parquet(preprocessed_path, index=False)

        kwargs["ti"].xcom_push(key="df_path", value=preprocessed_path)
        print(f"✅ Preprocessed data saved to {preprocessed_path}")

    def step2_model_prep(**kwargs):
        df_path = kwargs["ti"].xcom_pull(key="df_path", task_ids="preprocess")
        df = pd.read_parquet(df_path)

        X_processed = prepare_model_data(df)

        X_path = os.path.join(ARTIFACTS_DIR, "X_processed.joblib")
        joblib.dump(X_processed, X_path)

        kwargs["ti"].xcom_push(key="X_path", value=X_path)
        print(f"✅ Processed features saved to {X_path}")

    def step3_train(**kwargs):
        X_path = kwargs["ti"].xcom_pull(key="X_path", task_ids="model_prep")
        X_processed = joblib.load(X_path)

        models, X_test, y_test = run_anomaly_models(X_processed)

        models_path = os.path.join(ARTIFACTS_DIR, "models.joblib")
        X_test_path = os.path.join(ARTIFACTS_DIR, "X_test.joblib")
        y_test_path = os.path.join(ARTIFACTS_DIR, "y_test.joblib")

        joblib.dump(models, models_path)
        joblib.dump(X_test, X_test_path)
        joblib.dump(y_test, y_test_path)

        kwargs["ti"].xcom_push(key="models_path", value=models_path)
        kwargs["ti"].xcom_push(key="X_test_path", value=X_test_path)
        kwargs["ti"].xcom_push(key="y_test_path", value=y_test_path)

        print(f"✅ Models saved to {models_path}")

    def step4_evaluate(**kwargs):
        models_path = kwargs["ti"].xcom_pull(key="models_path", task_ids="train")
        X_test_path = kwargs["ti"].xcom_pull(key="X_test_path", task_ids="train")
        y_test_path = kwargs["ti"].xcom_pull(key="y_test_path", task_ids="train")

        models = joblib.load(models_path)
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)

        results = evaluate_models(models, X_test, y_test)

        results_path = os.path.join(ARTIFACTS_DIR, "results.csv")
        results.to_csv(results_path)

        print(f"✅ Evaluation results saved to {results_path}")

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=step1_preprocess,
    )

    model_prep_task = PythonOperator(
        task_id="model_prep",
        python_callable=step2_model_prep,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=step3_train,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=step4_evaluate,
    )

    preprocess_task >> model_prep_task >> train_task >> evaluate_task
