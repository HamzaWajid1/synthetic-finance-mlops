import pandas as pd


def clean_branches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the branches dataframe by handling:
    - Missing values (drop rows)
    - Duplicates
    - Typos in text fields
    - Inconsistent number/text formats
    - Non-sequential IDs
    """

    # -------------------------------
    # 1. Handle missing values
    # -------------------------------
    df["BranchName"] = df["BranchName"].fillna("Unknown")

    # Numeric fields
    df["BranchID"] = pd.to_numeric(df["BranchID"], errors="coerce")
    df["AddressID"] = pd.to_numeric(df["AddressID"], errors="coerce")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["BranchID", "AddressID"])

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Fix typos in text fields
    # -------------------------------
    # Example: normalize BranchName by stripping spaces
    df["BranchName"] = df["BranchName"].str.strip().str.title()

    # -------------------------------
    # 4. Fix inconsistent number/text formats
    # -------------------------------
    # Already handled above (numeric IDs)

    # -------------------------------
    # 5. Standardize date formats
    # -------------------------------
    # Not applicable

    # -------------------------------
    # 6. Check non-sequential IDs
    # -------------------------------
    if not df["BranchID"].is_unique:
        df = df.drop_duplicates(subset=["BranchID"], keep="first")
    df = df.sort_values("BranchID").reset_index(drop=True)

    # -------------------------------
    # 7. Future dates
    # -------------------------------
    # Not applicable

    return df


if __name__ == "__main__":
    # Example usage
    input_file = r"data\raw\branches.csv"
    output_file = r"data\processed\branches_cleaned.csv"

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    cleaned = clean_branches(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned branches to {output_file}")
