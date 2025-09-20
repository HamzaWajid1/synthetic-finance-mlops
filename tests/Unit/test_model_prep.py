import pandas as pd
import numpy as np
import pytest
from src.model_prep import prepare_model_data  # adjust path if needed

def test_prepare_model_data_robust():
    # --- Mock DataFrame ---
    df = pd.DataFrame({
        "TransactionTypeID": [1, 2],
        "Amount": [100, 200],
        "Origin_AccountTypeID": [10, 20],
        "Origin_AccountStatusID": [1, 2],
        "Origin_Balance": [1000, 2000],
        "Dest_AccountTypeID": [30, 40],
        "Dest_AccountStatusID": [1, 2],
        "Dest_Balance": [500, 1500],
        "Origin_CustomerTypeID": [1, 1],
        "Dest_CustomerTypeID": [2, 2],
        "Origin_LoanCount": [1, 2],
        "Origin_TotalPrincipal": [1000, 2000],
        "Origin_AvgInterestRate": [0.05, 0.06],
        "Dest_LoanCount": [1, 2],
        "Dest_TotalPrincipal": [500, 1500],
        "Dest_AvgInterestRate": [0.04, 0.05],
        "Origin_LoanStatus_Active": [1, 0],
        "Origin_LoanStatus_Overdue": [0, 1],
        "Origin_LoanStatus_Paid Off": [0, 1],
        "Dest_LoanStatus_Active": [1, 0],
        "Dest_LoanStatus_Overdue": [0, 1],
        "Dest_LoanStatus_Paid Off": [0, 1],
        "Amount_to_OriginBalance": [0.1, 0.15],
        "Amount_to_DestBalance": [0.2, 0.1333],
        "Amount_to_AvgTransaction": [0.15, 0.17],
        "Origin_AccountInactive": [0, 1],
        "Dest_AccountInactive": [1, 0],
        "Age_Difference": [5, 10],
        "Origin_LoanLeverage": [0.2, 0.3],
        "Dest_LoanLeverage": [0.1, 0.2],
        "TransactionHour": [12, 23],
        "TransactionWeekday": [1, 5],
        "TransactionMonth": [1, 12],
        "TransactionQuarter": [1, 4],
        "IsWeekend": [0, 1],
        "IsBusinessHours": [1, 0],
        "IsNightTime": [0, 1],
        "LargeTransferFlag": [0, 1],
        "VeryLargeTransferFlag": [0, 1],
        "UnusualTimingFlag": [1, 0],
        "HighRiskFlag": [0, 1],
        "CrossTypeTransfer": [1, 0]
    })

    features = df.columns.tolist()

    # --- Run function ---
    X_processed = prepare_model_data(df, features)

    # --- Assertions ---
    # Output shape
    assert X_processed.shape[0] == df.shape[0]

    numeric_features = [
        "Amount", "Origin_Balance", "Dest_Balance",
        "Origin_LoanCount", "Origin_TotalPrincipal", "Origin_AvgInterestRate",
        "Dest_LoanCount", "Dest_TotalPrincipal", "Dest_AvgInterestRate",
        "Origin_LoanStatus_Active", "Origin_LoanStatus_Overdue", "Origin_LoanStatus_Paid Off",
        "Dest_LoanStatus_Active", "Dest_LoanStatus_Overdue", "Dest_LoanStatus_Paid Off",
        "Amount_to_OriginBalance", "Amount_to_DestBalance", "Amount_to_AvgTransaction",
        "Age_Difference", "Origin_LoanLeverage", "Dest_LoanLeverage"
    ]

    # Numeric columns: first len(numeric_features) after ColumnTransformer
    numeric_data = X_processed[:, :len(numeric_features)]

    # Only test std for non-constant columns (avoid divide-by-zero)
    non_constant_mask = np.std(numeric_data, axis=0) != 0
    np.testing.assert_almost_equal(
        numeric_data[:, non_constant_mask].mean(axis=0),
        np.zeros(np.sum(non_constant_mask)),
        decimal=6
    )
    np.testing.assert_almost_equal(
        numeric_data[:, non_constant_mask].std(axis=0),
        np.ones(np.sum(non_constant_mask)),
        decimal=6
    )

    # Categorical columns: remaining columns
    cat_data = X_processed[:, len(numeric_features):]
    # Check that one-hot columns are 0/1 (after scaling they remain numeric but OneHotEncoder ensures 0/1)
    assert np.all((cat_data >= 0) & (cat_data <= 1))




