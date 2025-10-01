from .Utils.data_cleaner import *
from .Utils.data_merger import build_enriched_transactions
from .Utils.data_enricher import compute_transaction_features
import pandas as pd
import os


def preprocess_data(data_folder: str) -> pd.DataFrame:
    """
    Load raw CSV files, merge them into an enriched transactions DataFrame,
    clean inconsistencies, and compute additional transaction-level features.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing raw CSV files. 
        The folder must contain:
        - transactions.csv
        - transaction_types.csv
        - accounts.csv
        - account_types.csv
        - account_statuses.csv
        - customers.csv
        - customer_types.csv
        - addresses.csv
        - branches.csv
        - loans.csv
        - loan_statuses.csv

    Returns
    -------
    pd.DataFrame
        A cleaned + enriched DataFrame, ready for model preparation.
    """

    # === List of CSV names (must match filenames without .csv) ===
    csv_names = [
        "transactions", "transaction_types",
        "accounts", "account_types", "account_statuses",
        "customers", "customer_types", "addresses",
        "branches", "loans", "loan_statuses"
    ]

    # === Load CSVs into DataFrames ===
    dataframes = {}
    for name in csv_names:
        file_path = os.path.join(data_folder, f"{name}.csv")
        dataframes[name] = pd.read_csv(file_path)

    # === Clean raw DataFrames ===
    dataframes["accounts"] = clean_accounts(dataframes["accounts"])
    dataframes["addresses"] = clean_addresses(dataframes["addresses"])
    dataframes["branches"] = clean_branches(dataframes["branches"])
    dataframes["customers"] = clean_customers(dataframes["customers"])
    dataframes["loans"] = clean_loans(dataframes["loans"])

    # === Merge and enrich ===
    df = build_enriched_transactions(
        dataframes["transactions"], dataframes["transaction_types"],
        dataframes["accounts"], dataframes["account_types"], dataframes["account_statuses"],
        dataframes["customers"], dataframes["customer_types"], dataframes["addresses"],
        dataframes["branches"], dataframes["loans"], dataframes["loan_statuses"]
    )

    # === Add engineered features ===
    df = compute_transaction_features(df)

    # === Fill missing values ===
    df = df.fillna(0)

    #List of features to keep for model training.
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
    df = df[features_to_keep]

    print("✅ Preprocessing complete. Data ready for model prep.", flush=True)

    return df





