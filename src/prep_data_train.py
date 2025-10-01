from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

def run_anomaly_models(X_processed: pd.DataFrame):
    """
    Run anomaly detection + supervised models starting from processed features.

    Parameters
    ----------
    X_processed : pd.DataFrame
        Processed feature matrix for anomaly detection & supervised training.

    Returns
    -------
    models : dict
        Dictionary of trained models.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels for evaluation.
    """
    # ================================
    # 1. Anomaly detection
    # ================================
    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=500,
        contamination="auto",
        max_samples="auto",
        max_features=1.0,
        random_state=42
    )
    iso_forest.fit(X_processed)
    iso_label = (iso_forest.predict(X_processed) == -1).astype(int)

    # Local Outlier Factor
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.02,
        metric="euclidean",
        n_jobs=-1
    )
    lof_label = (lof.fit_predict(X_processed) == -1).astype(int)

    # Combine anomaly labels
    y = ((iso_label == 1) | (lof_label == 1)).astype(int)

    # ================================
    # 2. Train-test split
    # ================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )

    # ================================
    # 3. Supervised models
    # ================================
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
    }

    # Train models
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)

    return models, X_test, y_test
