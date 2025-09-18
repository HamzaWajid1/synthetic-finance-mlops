# tests/test_data_cleaning_extended.py
import pandas as pd
import pytest
from src.Utils.data_cleaner import clean_accounts, clean_addresses, clean_branches, clean_customers, clean_loans

# -------------------------------
# Edge cases for accounts
# -------------------------------
def test_clean_accounts_edge_cases():
    df = pd.DataFrame({
        "AccountID": ["1", "2", "abc", 4],
        "CustomerID": [100, "101", 102, None],
        "AccountTypeID": [1, 6, 3, 2],
        "AccountStatusID": [1, 2, 5, None],
        "Balance": ["1000", "NaN", None, "2000"],
        "OpeningDate": ["2020-01-01", "invalid-date", "2025-01-01", "2000-05-01"]
    })
    df_clean = clean_accounts(df)
    # Non-numeric IDs removed
    assert df_clean["AccountID"].dtype.kind in "iu"  # integer
    assert df_clean["CustomerID"].dtype.kind in "iu"
    # Invalid AccountTypeID/Status removed
    assert df_clean["AccountTypeID"].isin([1,2,3,4,5]).all()
    assert df_clean["AccountStatusID"].isin([1,2,3]).all()
    # Future dates removed
    assert (df_clean["OpeningDate"] <= pd.Timestamp.today()).all()

# -------------------------------
# Edge cases for addresses
# -------------------------------
def test_clean_addresses_edge_cases():
    df = pd.DataFrame({
        "AddressID": ["1", "2", None, 4],
        "Street": ["main st", "", None, "Broadway"],
        "City": ["new york", None, "los angeles", ""],
        "Country": ["usa", "US", "United States", None]
    })
    df_clean = clean_addresses(df)
    # Missing text replaced
    assert df_clean[["Street","City","Country"]].isna().sum().sum() == 0
    # IDs numeric & unique
    assert df_clean["AddressID"].dtype == int
    assert df_clean["AddressID"].is_unique

# -------------------------------
# Edge cases for branches
# -------------------------------
def test_clean_branches_edge_cases():
    df = pd.DataFrame({
        "BranchID": [1, "2", 2, None],
        "AddressID": [10, "11", 11, 12],
        "BranchName": [" Main ", "south ", None, "North"]
    })
    df_clean = clean_branches(df)
    # Names stripped and title-cased
    assert all(df_clean["BranchName"] == df_clean["BranchName"].str.title())
    # BranchID numeric & unique
    assert df_clean["BranchID"].is_unique

# -------------------------------
# Edge cases for customers
# -------------------------------
def test_clean_customers_edge_cases():
    df = pd.DataFrame({
        "CustomerID": [1, "2", 2, None],
        "FirstName": ["alice", None, "bob", "eve"],
        "LastName": ["smith", "jones", None, "doe"],
        "AddressID": [10, "11", None, 12],
        "CustomerTypeID": [1, 2, 5, 3],
        "DateOfBirth": ["2000-01-01", "invalid", "2025-01-01", None]
    })
    df_clean = clean_customers(df)
    # Only valid CustomerTypeID remain
    assert df_clean["CustomerTypeID"].isin([1,2,3]).all()
    # No future dates
    assert (df_clean["DateOfBirth"] <= pd.Timestamp.today()).all()

# -------------------------------
# Edge cases for loans
# -------------------------------
def test_clean_loans_edge_cases():
    df = pd.DataFrame({
        "LoanID": [1, "2", 2, None],
        "AccountID": [10, "11", 11, 12],
        "LoanStatusID": [1, 2, 5, 3],
        "PrincipalAmount": ["1000", "NaN", None, "2000"],
        "InterestRate": [0.05, 0.1, None, 0.15],
        "StartDate": ["2020-01-01", "invalid", "2025-01-01", None],
        "EstimatedEndDate": ["2025-01-01", "2030-01-01", None, "2028-01-01"]
    })
    df_clean = clean_loans(df)
    # Only valid LoanStatusID remain
    assert df_clean["LoanStatusID"].isin([1,2,3]).all()
    # Critical numeric fields no missing
    assert not df_clean[["LoanID","AccountID","PrincipalAmount","InterestRate"]].isna().any().any()
    # Future dates removed
    assert (df_clean["StartDate"] <= pd.Timestamp.today()).all()
