import pandas as pd
from rapidfuzz import process


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions dataframe by handling:
    - Missing values
    - Duplicates
    - Inconsistent number/text formats
    - Mixed date formats
    - Non-sequential IDs
    - Future dates
    """

    # -------------------------------
    # 1. Handle missing values
    # -------------------------------
    # Amount and IDs must be numeric
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["AccountOriginID"] = pd.to_numeric(df["AccountOriginID"], errors="coerce")
    df["AccountDestinationID"] = pd.to_numeric(df["AccountDestinationID"], errors="coerce")
    df["BranchID"] = pd.to_numeric(df["BranchID"], errors="coerce")
    df["TransactionTypeID"] = pd.to_numeric(df["TransactionTypeID"], errors="coerce")

    # Fill missing Description
    df["Description"] = df["Description"].fillna("Unknown")

    # Drop rows with critical missing data (IDs or Amount)
    df = df.dropna(subset=["TransactionID", "AccountOriginID", "AccountDestinationID", "Amount"])

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Fix typos (example: TransactionType ID mapping)
    # -------------------------------
    # Suppose TransactionTypeID must map to {1, 2, 3,4}
    valid_types = [1, 2, 3, 4]
    df = df[df["TransactionTypeID"].isin(valid_types)]
    
    # Normalize amounts with commas or 'k' notation
    def normalize_amount(x):
        if isinstance(x, str):
            x = x.replace(",", "").lower()
            if "k" in x: 
                x = float(x.replace("k", "")) * 1000
        return float(x)
    
    df["Amount"] = df["Amount"].apply(normalize_amount)

    # -------------------------------
    # 4. Standardize date formats
    # -------------------------------
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

    # -------------------------------
    # 5. Check non-sequential IDs
    # -------------------------------
    if not df["TransactionID"].is_unique:
        df = df.drop_duplicates(subset=["TransactionID"], keep="first")
    df = df.sort_values("TransactionID").reset_index(drop=True)

    # -------------------------------
    # 6. Remove future dates
    # -------------------------------
    today = pd.Timestamp.today()
    df = df[df["TransactionDate"] <= today]

    return df


if __name__ == "__main__":
    # Example usage
    input_file = r"data\raw\transactions.csv"
    output_file = r"data\processed\transactions_cleaned.csv"

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    cleaned = clean_transactions(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned transactions to {output_file}")
