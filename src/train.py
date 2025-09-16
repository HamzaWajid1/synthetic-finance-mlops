from src.preprocess import preprocess_data
from src.model_prep import prepare_model_data
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, classification_report,
    roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb
import pandas as pd
#import lightgbm as lgb
#from catboost import CatBoostClassifier

def run_anomaly_supervised_pipeline(data_folder: str):
    """
    Full anomaly detection + supervised model training pipeline.

    Parameters
    ----------
    data_folder : str
        Path to folder containing raw data CSVs.

    Returns
    -------
    models : dict
        Dictionary containing trained models. Keys are model names, values are model objects.
    X_test : pd.DataFrame
        Test features to evaluate the models on.
    y_test : pd.Series
        True labels corresponding to X_test.
    """
    # ================================
    # 1. Load & preprocess data
    # ================================
    df = preprocess_data(data_folder)

    features_to_keep = [
        "TransactionTypeID", "Amount",
        "Origin_AccountTypeID", "Origin_AccountStatusID", "Origin_Balance",
        "Dest_AccountTypeID", "Dest_AccountStatusID", "Dest_Balance",
        "Origin_CustomerTypeID", "Dest_CustomerTypeID",
        "Origin_LoanCount", "Origin_TotalPrincipal", "Origin_AvgInterestRate",
        "Dest_LoanCount", "Dest_TotalPrincipal", "Dest_AvgInterestRate",
        "Origin_LoanStatus_Active", "Origin_LoanStatus_Overdue", "Origin_LoanStatus_Paid Off",
        "Dest_LoanStatus_Active", "Dest_LoanStatus_Overdue", "Dest_LoanStatus_Paid Off",
        "Amount_to_OriginBalance", "Amount_to_DestBalance", "Amount_to_AvgTransaction",
        "Origin_AccountInactive", "Dest_AccountInactive", "Age_Difference",
        "Origin_LoanLeverage", "Dest_LoanLeverage",
        "TransactionHour", "TransactionWeekday", "TransactionMonth", "TransactionQuarter",
        "IsWeekend", "IsBusinessHours", "IsNightTime",
        "LargeTransferFlag", "VeryLargeTransferFlag", "UnusualTimingFlag", "HighRiskFlag", "CrossTypeTransfer"
    ]

    X_processed = prepare_model_data(df, features_to_keep)

    # ================================
    # 2. Anomaly detection
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

    # Add anomaly labels to original df
    df['anomaly_lof'] = lof_label
    df['anomaly_iso'] = iso_label
    df['anomaly_label'] = ((df['anomaly_iso'] == 1) | (df['anomaly_lof'] == 1)).astype(int)

    # ================================
    # 3. Train-test split
    # ================================
    y = df['anomaly_label']
    X = df.drop(columns=['anomaly_label', 'anomaly_iso', 'anomaly_lof'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ================================
    # 4. Define models
    # ================================
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
        #"LightGBM": lgb.LGBMClassifier(
        #    n_estimators=500, learning_rate=0.05, 
        #    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), 
        #    random_state=42
        #   ),
        #"CatBoost": CatBoostClassifier(
        #    iterations=500, learning_rate=0.05, depth=6, 
        #    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), 
        #    verbose=0, random_state=42
        #)    
    }

    # ================================
    # 5. Train
    # ================================
    results = []

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
    
    # return trained models with X_test,y_test for evaluation
    return models,X_test,y_test
        
