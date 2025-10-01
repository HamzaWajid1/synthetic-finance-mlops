from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects useful features, standardizes them, and returns scaled data.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (from preprocess.py).
    

    Returns
    -------
    pd.DataFrame
        Scaled feature DataFrame, ready for unsupervised model training.
    """
    # Subset features
    df_selected = df.copy()

    # Numerical features to scale
    numeric_features = [
        "Amount", "Origin_Balance", "Dest_Balance", 
        "Origin_LoanCount", "Origin_TotalPrincipal", "Origin_AvgInterestRate",
        "Dest_LoanCount", "Dest_TotalPrincipal", "Dest_AvgInterestRate",
        "Origin_LoanStatus_Active", "Origin_LoanStatus_Overdue", "Origin_LoanStatus_Paid Off",
        "Dest_LoanStatus_Active", "Dest_LoanStatus_Overdue", "Dest_LoanStatus_Paid Off",
        "Amount_to_OriginBalance", "Amount_to_DestBalance", "Amount_to_AvgTransaction",
        "Age_Difference", "Origin_LoanLeverage", "Dest_LoanLeverage"
    ]

    # Define features from DataFrame
    features = df_selected.columns.tolist()
    
    # All remaining features → categorical (IDs + binary flags)
    categorical_features = [
        col for col in features if col not in numeric_features
    ]

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
    ]
    )

    X_processed = preprocessor.fit_transform(df_selected)

    print("✅ Model data prepared (scaled).", flush=True)
    return X_processed
