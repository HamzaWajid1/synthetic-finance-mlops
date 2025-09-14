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
