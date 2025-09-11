import pandas as pd
from rapidfuzz import process


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the customers dataframe by handling:
    - Missing values
    - Duplicates
    - Typos in categorical fields
    - Inconsistent formats
    - Mixed date formats
    - Non-sequential IDs
    - Future dates
    """

    # -------------------------------
    # 1. Handle missing values
    # -------------------------------   
    df["FirstName"] = df["FirstName"].fillna("Unknown")
    df["LastName"] = df["LastName"].fillna("Unknown")
    df["AddressID"] = df["AddressID"].fillna(0)
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")
    df["DateOfBirth"].fillna(pd.Timestamp("1970-01-01"))

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Fix typos (example: CustomerTypeID mapping)
    # -------------------------------
    # Suppose CustomerTypeID must map to {1, 2, 3}
    valid_types = [1, 2, 3]
    df = df[df["CustomerTypeID"].isin(valid_types)]

    # Example for country (if customers table had country info)
    # valid_countries = ["United States"]
    # df["Country"] = df["Country"].apply(
    #     lambda x: process.extractOne(x, valid_countries)[0]
    # )

    # -------------------------------
    # 4. Fix inconsistent formats
    # -------------------------------
    # Ensure AddressID is numeric
    df["AddressID"] = pd.to_numeric(df["AddressID"], errors="coerce")

    # -------------------------------
    # 5. Standardize date formats
    # -------------------------------
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")

    # -------------------------------
    # 6. Check non-sequential IDs
    # -------------------------------
    if not df["CustomerID"].is_unique:
        df = df.drop_duplicates(subset=["CustomerID"], keep="first")

    # -------------------------------
    # 7. Remove future dates
    # -------------------------------
    today = pd.Timestamp.today()
    df = df[df["DateOfBirth"] <= today]

    return df


if __name__ == "__main__":
    # Example usage
    input_file = r"data\raw\customers.csv"
    output_file = r"data\processed\customers_cleaned.csv"

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    cleaned = clean_customers(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned customers to {output_file}")
