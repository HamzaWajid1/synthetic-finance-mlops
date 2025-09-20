import pandas as pd
import pytest
from src.Utils.data_enricher import compute_transaction_features


@pytest.fixture
def enriched_transactions_mock():
    """Creates a small mock DataFrame simulating the enriched transactions."""
    return pd.DataFrame({
        "TransactionTypeID": [1, 2],
        "Amount": [1000, 500],

        # Required ID fields
        "Origin_AccountTypeID": [1, 2],
        "Origin_AccountStatusID": [1, 3],
        "Dest_AccountTypeID": [2, 3],
        "Dest_AccountStatusID": [2, 3],
        "Origin_CustomerTypeID": [1, 3],
        "Dest_CustomerTypeID": [1, 2],

        # Human-readable account info
        "Origin_AccountType": ["Checking", "Savings"],
        "Dest_AccountType": ["Savings", "Checking"],
        "Origin_AccountStatus": ["Active", "Inactive"],
        "Dest_AccountStatus": ["Closed", "Active"],

        # Balances
        "Origin_Balance": [2000, 0],
        "Dest_Balance": [5000, 1000],

        # Customer info
        "Origin_DateOfBirth": ["1980-01-01", "1990-06-15"],
        "Dest_DateOfBirth": ["1975-05-20", "1985-03-10"],
        "Origin_TypeName": ["Individual", "Small Business"],
        "Dest_TypeName": ["Individual", "Individual"],

        # Loan info
        "Origin_TotalPrincipal": [1000, 0],
        "Dest_TotalPrincipal": [500, 0],
        "Origin_LoanCount": [1, 0],
        "Dest_LoanCount": [1, 0],
        "Origin_AvgInterestRate": [0.05, 0.12],
        "Dest_AvgInterestRate": [0.04, 0.15],
        "Origin_LoanStatus_Active": [1, 0],
        "Origin_LoanStatus_Overdue": [0, 0],
        "Origin_LoanStatus_Paid Off": [0, 0],
        "Dest_LoanStatus_Active": [1, 0],
        "Dest_LoanStatus_Overdue": [0, 0],
        "Dest_LoanStatus_Paid Off": [0, 0],

        # Time
        "TransactionDate": ["2025-09-18 14:00:00", "2025-09-17 23:00:00"]
    })


def test_feature_columns_exist(enriched_transactions_mock):
    """Test that all expected engineered features are created."""
    df_features = compute_transaction_features(enriched_transactions_mock)

    expected_columns =[
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

    for col in expected_columns:
        assert col in df_features.columns


def test_no_nans(enriched_transactions_mock):
    """Ensure all NaN values are filled with 0."""
    df_features = compute_transaction_features(enriched_transactions_mock)
    assert df_features.isnull().sum().sum() == 0


def test_computed_values(enriched_transactions_mock):
    """Check that some features are calculated correctly."""
    df_features = compute_transaction_features(enriched_transactions_mock)

    # Amount ratios
    assert df_features.loc[0, "Amount_to_OriginBalance"] == 1000 / 2000
    assert df_features.loc[1, "Amount_to_OriginBalance"] == 0  # division by zero handled

    # Account inactivity flags
    assert df_features.loc[0, "Origin_AccountInactive"] == 0
    assert df_features.loc[1, "Origin_AccountInactive"] == 1

    # High risk flag logic
    assert df_features.loc[0, "HighRiskFlag"] == 1
    assert df_features.loc[1, "HighRiskFlag"] == 1

    # Cross type transfer
    assert df_features.loc[0, "CrossTypeTransfer"] == 0
    assert df_features.loc[1, "CrossTypeTransfer"] == 1

    # Night time flag
    assert df_features.loc[1, "IsNightTime"] == 1


def test_large_transfer_flags(enriched_transactions_mock):
    """Test LargeTransferFlag and VeryLargeTransferFlag."""
    df_features = compute_transaction_features(enriched_transactions_mock)

    # Transaction 0: 1000 / 2000 = 0.5 → LargeTransferFlag = 0, VeryLargeTransferFlag = 0
    assert df_features.loc[0, "LargeTransferFlag"] == 0
    assert df_features.loc[0, "VeryLargeTransferFlag"] == 0

    # Transaction 1: Amount_to_OriginBalance = 0 → both flags 0
    assert df_features.loc[1, "LargeTransferFlag"] == 0
    assert df_features.loc[1, "VeryLargeTransferFlag"] == 0


