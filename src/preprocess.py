from Utils.data_cleaner import *
from Utils.data_merger import build_enriched_transactions
from Utils.data_enricher import compute_transaction_features

import pandas as pd
import os


def preprocess_data(data_folder: str) -> pd.DataFrame:
    """
    Load raw CSV files, merge them into an enriched transactions DataFrame,
    and compute additional transaction-level features.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing raw CSV files. 
        The folder must contain the following files:
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
        A DataFrame containing enriched and feature-engineered transactions.
    """

    # === List of CSV names (must match filenames without .csv) ===
    csv_names = [
        "transactions", "transaction_types",
        "accounts", "account_types", "account_statuses",
        "customers", "customer_types", "addresses",
        "branches", "loans", "loan_statuses"
    ]

    # === Load each CSV into a DataFrame with the same variable name ===
    dataframes = {}
    for name in csv_names:
        file_path = os.path.join(data_folder, f"{name}.csv")
        dataframes[name] = pd.read_csv(file_path)

    # === Merge and enrich transactions ===
    df = build_enriched_transactions(
        dataframes["transactions"], dataframes["transaction_types"],
        dataframes["accounts"], dataframes["account_types"], dataframes["account_statuses"],
        dataframes["customers"], dataframes["customer_types"], dataframes["addresses"],
        dataframes["branches"], dataframes["loans"], dataframes["loan_statuses"]
    )

    # === Add additional transaction-level features ===
    df = compute_transaction_features(df)

    return df




