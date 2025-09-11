import pandas as pd


def clean_loans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the loans dataframe by handling:
    - Missing values (drop rows)
    - Duplicates
    - Inconsistent number/text formats
    - Mixed date formats
    - Non-sequential IDs
    - Future dates
    - Typos in categorical fields (LoanStatusID)
    """

    # -------------------------------
    # 1. Handle missing values
    # -------------------------------
    # Numeric fields
    df["PrincipalAmount"] = pd.to_numeric(df["PrincipalAmount"], errors="coerce")
    df["InterestRate"] = pd.to_numeric(df["InterestRate"], errors="coerce")
    df["LoanID"] = pd.to_numeric(df["LoanID"], errors="coerce")
    df["AccountID"] = pd.to_numeric(df["AccountID"], errors="coerce")
    df["LoanStatusID"] = pd.to_numeric(df["LoanStatusID"], errors="coerce")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["LoanID", "AccountID", "LoanStatusID", "PrincipalAmount", "InterestRate"])

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Fix typos in categorical fields
    # -------------------------------
    # Valid LoanStatusIDs = [1, 2, 3]
    valid_statuses = [1, 2, 3]
    df = df[df["LoanStatusID"].isin(valid_statuses)]

    # -------------------------------
    # 4. Fix inconsistent number/text formats
    # -------------------------------
    # Already ensured numeric conversion above

    # -------------------------------
    # 5. Standardize date formats
    # -------------------------------
    # Dates
    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
    df["EstimatedEndDate"] = pd.to_datetime(df["EstimatedEndDate"], errors="coerce")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["StartDate", "EstimatedEndDate"])

    # -------------------------------
    # 6. Check non-sequential IDs
    # -------------------------------
    if not df["LoanID"].is_unique:
        df = df.drop_duplicates(subset=["LoanID"], keep="first")
    df = df.sort_values("LoanID").reset_index(drop=True)

    # -------------------------------
    # 7. Remove future dates
    # -------------------------------
    today = pd.Timestamp.today()
    df = df[df["StartDate"] <= today]
    
    return df


if __name__ == "__main__":
    # Example usage
    input_file = r"data\raw\loans.csv"
    output_file = r"data\processed\loans_cleaned.csv"

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    cleaned = clean_loans(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned loans to {output_file}")
