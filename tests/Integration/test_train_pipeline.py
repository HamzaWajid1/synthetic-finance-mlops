import pytest
import pandas as pd
from src.train import run_anomaly_supervised_pipeline

#@pytest.mark.skip(reason="Integration test using full original dataset; run manually")
def test_run_anomaly_supervised_pipeline_with_real_data():
    """
    End-to-end integration test for the full anomaly detection
    + supervised training pipeline using the original dataset.
    """

    # Path to the folder containing all raw CSVs
    data_folder = r"C:\Users\hamza\OneDrive\Desktop\InterviewPrepUSA\UCSC_Extension\IntroToMachineLearning\synthetic-finance-mlops\data\raw"

    # Run full pipeline
    models, X_test, y_test = run_anomaly_supervised_pipeline(data_folder)

    # --- Assertions ---
    # Models dictionary
    assert isinstance(models, dict)
    assert len(models) > 0
    for model in models.values():
        assert hasattr(model, "predict")  # ensure trained model has predict method

    # Test set checks
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert X_test.shape[0] == y_test.shape[0]
    assert X_test.shape[1] > 0

    # Optionally: check that anomaly label exists
    assert 'anomaly_label' not in X_test.columns  # should be dropped
