import pandas as pd
from rapidfuzz import process


def clean_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the accounts dataframe by handling:
    - Missing values
    - Duplicates
    - Typos in categorical fields
    - Inconsistent number/text formats
    - Mixed date formats
    - Non-sequential IDs
    - Future dates
    """

    # -------------------------------
    # 1. Handle missing values
    # -------------------------------
    # Numeric fields: Balance
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")
    #df["Balance"].fillna(df["Balance"].median(), inplace=True)

    # Categorical IDs: AccountTypeID, AccountStatusID
    df["AccountTypeID"] = pd.to_numeric(df["AccountTypeID"], errors="coerce")
    df["AccountStatusID"] = pd.to_numeric(df["AccountStatusID"], errors="coerce")

    # Dates
    df["OpeningDate"] = pd.to_datetime(df["OpeningDate"], errors="coerce")
    df["OpeningDate"] = df['OpeningDate'].fillna(pd.Timestamp("2000-01-01"))
    
    # Drop rows with critical missing data (IDs or Amount)
    df = df.dropna(subset=["AccountID", "CustomerID", "AccountTypeID", "AccountStatusID", "Balance"])

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Fix typos in categorical fields
    # -------------------------------
    # Suppose valid AccountTypeIDs = [1,2,3,4,5] and AccountStatusIDs = [1,2,3]
    valid_account_types = [1, 2, 3, 4, 5]
    valid_statuses = [1, 2, 3]

    df = df[df["AccountTypeID"].isin(valid_account_types)]
    df = df[df["AccountStatusID"].isin(valid_statuses)]

    # -------------------------------
    # 4. Fix inconsistent number/text formats
    # -------------------------------
    # Ensure AccountID and CustomerID are numeric
    df["AccountID"] = pd.to_numeric(df["AccountID"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

    # -------------------------------
    # 5. Standardize date formats
    # -------------------------------
    df["OpeningDate"] = pd.to_datetime(df["OpeningDate"], errors="coerce")

    # -------------------------------
    # 6. Check non-sequential IDs
    # -------------------------------
    if not df["AccountID"].is_unique:
        df = df.drop_duplicates(subset=["AccountID"], keep="first")

    df = df.sort_values("AccountID").reset_index(drop=True)

    # -------------------------------
    # 7. Remove future dates
    # -------------------------------
    today = pd.Timestamp.today()
    df = df[df["OpeningDate"] <= today]
    
    return df


if __name__ == "__main__":
    # Example usage
    input_file = r"data\raw\accounts.csv"
    output_file = r"data\processed\accounts_cleaned.csv"

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    cleaned = clean_accounts(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned accounts to {output_file}")
