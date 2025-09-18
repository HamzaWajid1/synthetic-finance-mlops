# tests/test_data_cleaning.py
import pandas as pd
import pytest
from src.Utils.data_cleaner import clean_accounts, clean_addresses, clean_branches, clean_customers, clean_loans

# -------------------------------
# Sample input data for tests
# -------------------------------
@pytest.fixture
def accounts_df():
    return pd.DataFrame({
        "AccountID": [1, 2, 2, None],
        "CustomerID": [100, 101, 101, 102],
        "AccountTypeID": [1, 5, 5, 6],
        "AccountStatusID": [1, 3, 3, None],
        "Balance": ["1000", "2000", "2000", "NaN"],
        "OpeningDate": ["2020-01-01", "2023-01-01", "2023-01-01", "2025-01-01"]
    })

@pytest.fixture
def addresses_df():
    return pd.DataFrame({
        "AddressID": [1, 2, 2, None],
        "Street": ["Main St", None, "Main St", "Broadway"],
        "City": ["New York", "Los Angeles", "Los Angeles", None],
        "Country": ["United States", "US", "USA", None]
    })

# -------------------------------
# Accounts tests
# -------------------------------
def test_clean_accounts_drops_invalid(accounts_df):
    df_clean = clean_accounts(accounts_df)
    # Rows with invalid AccountTypeID or AccountStatusID removed
    assert df_clean["AccountTypeID"].isin([1,2,3,4,5]).all()
    assert df_clean["AccountStatusID"].isin([1,2,3]).all()
    # Check no missing values in critical columns
    assert not df_clean[["AccountID", "CustomerID", "Balance"]].isna().any().any()
    # Check duplicates removed
    assert df_clean["AccountID"].is_unique
    # Check future dates removed
    assert (df_clean["OpeningDate"] <= pd.Timestamp.today()).all()

# -------------------------------
# Addresses tests
# -------------------------------
def test_clean_addresses_fill_missing(addresses_df):
    df_clean = clean_addresses(addresses_df)
    # Missing Street/City/Country replaced
    assert df_clean["Street"].isna().sum() == 0
    assert df_clean["City"].isna().sum() == 0
    assert df_clean["Country"].isna().sum() == 0
    # AddressID numeric & unique
    assert df_clean["AddressID"].dtype == int
    assert df_clean["AddressID"].is_unique

# -------------------------------
# More tests for branches, customers, loans
# -------------------------------
def test_clean_branches_basic():
    df = pd.DataFrame({
        "BranchID": [1, 2, 2],
        "AddressID": [10, 11, 11],
        "BranchName": ["Main", None, "Main"]
    })
    df_clean = clean_branches(df)
    assert df_clean["BranchID"].is_unique
    assert df_clean["BranchName"].notna().all()

def test_clean_customers_basic():
    df = pd.DataFrame({
        "CustomerID": [1, 2, 2],
        "FirstName": ["Alice", None, "Bob"],
        "LastName": ["Smith", "Jones", None],
        "AddressID": [10, 11, 11],
        "CustomerTypeID": [1, 2, 5],
        "DateOfBirth": ["2000-01-01", "2025-01-01", None]
    })
    df_clean = clean_customers(df)
    # Check type filter
    assert df_clean["CustomerTypeID"].isin([1,2,3]).all()
    # Check future dates removed
    assert (df_clean["DateOfBirth"] <= pd.Timestamp.today()).all()

def test_clean_loans_basic():
    df = pd.DataFrame({
        "LoanID": [1, 2, 2],
        "AccountID": [10, 11, 11],
        "LoanStatusID": [1, 2, 5],
        "PrincipalAmount": ["1000", "2000", "3000"],
        "InterestRate": [0.05, 0.1, 0.15],
        "StartDate": ["2020-01-01", "2025-01-01", "2023-01-01"],
        "EstimatedEndDate": ["2025-01-01", "2030-01-01", None]
    })
    df_clean = clean_loans(df)
    # Only valid LoanStatusID remain
    assert df_clean["LoanStatusID"].isin([1,2,3]).all()
    # No missing critical fields
    assert not df_clean[["LoanID","AccountID","PrincipalAmount"]].isna().any().any()
    # Future dates removed
    assert (df_clean["StartDate"] <= pd.Timestamp.today()).all()
