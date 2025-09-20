import pandas as pd
import pytest
from src.Utils.data_merger import build_enriched_transactions

# ===================== Fixtures =====================

@pytest.fixture
def transactions():
    return pd.DataFrame({
        "TransactionID": [3022681, 3037846],
        "AccountOriginID": [201164, 200138],
        "AccountDestinationID": [200868, 201402],
        "TransactionTypeID": [2, 2],
        "Amount": [855.17, 806.2],
        "TransactionDate": pd.to_datetime(["2023-04-20", "2021-08-10"]),
        "BranchID": [41, 43],
        "Description": ["Transaction 22681", "Transaction 37846"]
    })

@pytest.fixture
def transaction_types():
    return pd.DataFrame({
        "TransactionTypeID": [1, 2, 3, 4],
        "TypeName": ["Deposit", "Withdrawal", "Transfer", "Payment"]
    })

@pytest.fixture
def accounts():
    return pd.DataFrame({
        "AccountID": [200868, 201164],
        "CustomerID": [10123, 10124],
        "AccountTypeID": [3, 1],
        "AccountStatusID": [1, 2],
        "Balance": [48348.54, 35001.41],
        "OpeningDate": pd.to_datetime(["2018-06-12", "2019-10-30"])
    })

@pytest.fixture
def account_types():
    return pd.DataFrame({
        "AccountTypeID": [1, 3],
        "TypeName": ["Checking", "Payroll"]
    })

@pytest.fixture
def account_statuses():
    return pd.DataFrame({
        "AccountStatusID": [1, 2],
        "StatusName": ["Active", "Inactive"]
    })

@pytest.fixture
def customers():
    return pd.DataFrame({
        "CustomerID": [10123, 10124],
        "FirstName": ["John", "Alice"],
        "LastName": ["Doe", "Smith"],
        "DateOfBirth": pd.to_datetime(["1980-01-01", "1990-02-02"]),
        "AddressID": [706, 707],
        "CustomerTypeID": [1, 2]
    })

@pytest.fixture
def customer_types():
    return pd.DataFrame({
        "CustomerTypeID": [1, 2],
        "TypeName": ["Individual", "Small Business"]
    })

@pytest.fixture
def addresses():
    return pd.DataFrame({
        "AddressID": [706, 707],
        "Street": ["Edgardo", "Fernwood"],
        "City": ["Stafford", "Opelousas"],
        "Country": ["United States", "United States"]
    })

@pytest.fixture
def branches():
    return pd.DataFrame({
        "BranchID": [41, 43],
        "BranchName": ["Branch 1", "Branch 2"],
        "AddressID": [733, 511]
    })

@pytest.fixture
def loans():
    return pd.DataFrame({
        "LoanID": [400230, 400307],
        "AccountID": [200868, 201164],
        "LoanStatusID": [1, 2],
        "PrincipalAmount": [76958.56, 29013.67],
        "InterestRate": [0.0547, 0.0321],
        "StartDate": pd.to_datetime(["2022-11-20", "2022-02-22"]),
        "EstimatedEndDate": pd.to_datetime(["2026-08-06", "2025-12-08"])
    })

@pytest.fixture
def loan_statuses():
    return pd.DataFrame({
        "LoanStatusID": [1, 2],
        "StatusName": ["Active", "Paid Off"]
    })

# ===================== Tests =====================

def test_enriched_transactions(transactions, transaction_types, accounts, account_types,
                               account_statuses, customers, customer_types, addresses,
                               branches, loans, loan_statuses):

    enriched = build_enriched_transactions(
        transactions, transaction_types, accounts, account_types, account_statuses,
        customers, customer_types, addresses, branches, loans, loan_statuses
    )

    # Check main columns exist
    assert "TransactionTypeName" in enriched.columns
    assert "Origin_AccountType" in enriched.columns
    assert "Dest_AccountStatus" in enriched.columns
    assert "Origin_LoanCount" in enriched.columns
    assert "Dest_LoanCount" in enriched.columns

    # Check row count matches input transactions
    assert len(enriched) == len(transactions)
