import pandas as pd
import os
from src.preprocess import preprocess_data
from src.model_prep import prepare_model_data
from src.prep_data_train import run_anomaly_models
from src.evaluate import evaluate_models

def main():
    """
    Main pipeline for synthetic finance anomaly detection.
    """
    try:
        # =========================
        # 1. Load sample data
        # =========================
        print("📌 Step 1: Loading and preprocessing data")
        data_folder = "data/raw"
        
        # Check if data folder exists
        if not os.path.exists(data_folder):
            print(f"❌ Error: Data folder '{data_folder}' not found!")
            print("Please ensure the data folder exists with the required CSV files.")
            return
        
        # Preprocess data
        df_preprocessed = preprocess_data(data_folder)
        print(f"✅ Preprocessed data shape: {df_preprocessed.shape}")

        # =========================
        # 2. Prepare model data
        # =========================
        print("\n📌 Step 2: Preparing model data")
        X_processed = prepare_model_data(df_preprocessed)
        print(f"✅ Model data shape: {X_processed.shape}")

        # =========================
        # 3. Train models
        # =========================
        print("\n📌 Step 3: Training anomaly detection models")
        models, X_test, y_test = run_anomaly_models(X_processed)
        print(f"✅ Trained {len(models)} models")

        # =========================
        # 4. Evaluate models
        # =========================
        print("\n📌 Step 4: Evaluating models")
        results = evaluate_models(models, X_test, y_test)

        print("\n✅ Final Results:")
        print(results)
        
        return results

    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
