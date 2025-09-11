import pandas as pd
from rapidfuzz import process


def clean_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the addresses dataframe by handling:
    - Missing values (drop rows)
    - Duplicates
    - Typos in text fields
    - Inconsistent text formats
    - Non-sequential IDs
    """

    # -------------------------------
    # 1. Handle missing values
    # -------------------------------
    df["Street"] = df["Street"].fillna("Unknown")
    df["City"] = df["City"].fillna("Unknown")
    df["Country"] = df["Country"].fillna("Unknown")

    # Drop rows with missing AddressID
    df["AddressID"] = pd.to_numeric(df["AddressID"], errors="coerce")
    df = df.dropna(subset=["AddressID"])

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Fix typos in text fields
    # -------------------------------
    # Example: standardize Country field
    valid_countries = ["United States"]  # expand if needed
    df["Country"] = df["Country"].apply(
        lambda x: process.extractOne(str(x), valid_countries)[0]
    )

    # -------------------------------
    # 4. Fix inconsistent text formats
    # -------------------------------
    df["Street"] = df["Street"].str.title().str.strip()
    df["City"] = df["City"].str.title().str.strip()
    df["Country"] = df["Country"].str.title().str.strip()

    # -------------------------------
    # 5. Standardize ID format
    # -------------------------------
    df["AddressID"] = df["AddressID"].astype(int)

    # -------------------------------
    # 6. Check non-sequential IDs
    # -------------------------------
    if not df["AddressID"].is_unique:
        df = df.drop_duplicates(subset=["AddressID"], keep="first")

    df = df.sort_values("AddressID").reset_index(drop=True)

    # -------------------------------
    # 7. Future dates
    # -------------------------------
    # Not applicable for Addresses table

    return df


if __name__ == "__main__":
    # Example usage
    input_file = r"data\raw\addresses.csv"
    output_file = r"data\processed\addresses_cleaned.csv"

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    cleaned = clean_addresses(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned addresses to {output_file}")
