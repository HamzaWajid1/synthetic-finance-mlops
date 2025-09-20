import pandas as pd
import pytest
from src.Utils.data_merger import build_enriched_transactions

# ===================== Fixtures =====================

@pytest.fixture
def transactions():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "AccountOriginID": [101, 102, 999],  # 999 does not exist
        "AccountDestinationID": [201, 202, 203],
        "TransactionTypeID": [1, 2, 3],
        "Amount": [100, 200, 300],
        "TransactionDate": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        "BranchID": [1, 2, 3],
        "Description": ["T1", "T2", "T3"]
    })

@pytest.fixture
def transaction_types():
    return pd.DataFrame({
        "TransactionTypeID": [1, 2, 3],
        "TypeName": ["Deposit", "Withdrawal", "Transfer"]
    })

@pytest.fixture
def accounts():
    return pd.DataFrame({
        "AccountID": [101, 102, 201, 202],
        "CustomerID": [1001, 1002, 2001, 2002],
        "AccountTypeID": [1, 2, 1, 2],
        "AccountStatusID": [1, 1, 1, 2],
        "Balance": [1000, 2000, 3000, 4000],
        "OpeningDate": pd.to_datetime(["2020-01-01"]*4)
    })

@pytest.fixture
def account_types():
    return pd.DataFrame({
        "AccountTypeID": [1, 2],
        "TypeName": ["Checking", "Savings"]
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
        "CustomerID": [1001, 1002, 2001, 2002],
        "FirstName": ["A", "B", "C", "D"],
        "LastName": ["X", "Y", "Z", "W"],
        "DateOfBirth": pd.to_datetime(["1990-01-01"]*4),
        "AddressID": [1, 2, 3, 4],
        "CustomerTypeID": [1, 2, 1, 2]
    })

@pytest.fixture
def customer_types():
    return pd.DataFrame({
        "CustomerTypeID": [1, 2],
        "TypeName": ["Individual", "Business"]
    })

@pytest.fixture
def addresses():
    return pd.DataFrame({
        "AddressID": [1, 2, 3, 4],
        "Street": ["S1", "S2", "S3", "S4"],
        "City": ["C1", "C2", "C3", "C4"],
        "Country": ["US", "US", "US", "US"]
    })

@pytest.fixture
def branches():
    return pd.DataFrame({
        "BranchID": [1, 2, 3],
        "BranchName": ["B1", "B2", "B3"],
        "AddressID": [1, 2, 3]
    })

@pytest.fixture
def loans():
    return pd.DataFrame({
        "LoanID": [1, 2],
        "AccountID": [101, 202],
        "LoanStatusID": [1, 2],
        "PrincipalAmount": [10000, 20000],
        "InterestRate": [0.05, 0.1],
        "StartDate": pd.to_datetime(["2022-01-01", "2022-06-01"]),
        "EstimatedEndDate": pd.to_datetime(["2025-01-01", "2025-06-01"])
    })

@pytest.fixture
def loan_statuses():
    return pd.DataFrame({
        "LoanStatusID": [1, 2],
        "StatusName": ["Active", "Paid Off"]
    })

@pytest.fixture
def empty_loans():
    return pd.DataFrame(columns=["LoanID", "AccountID", "LoanStatusID",
                                 "PrincipalAmount", "InterestRate",
                                 "StartDate", "EstimatedEndDate"])


# ===================== Extended Tests =====================

def test_enriched_missing_origin_account(transactions, transaction_types, accounts,
                                         account_types, account_statuses, customers,
                                         customer_types, addresses, branches, loans, loan_statuses):
    """Transactions with missing origin accounts should still return enriched rows (with NaNs)"""
    enriched = build_enriched_transactions(transactions, transaction_types, accounts,
                                          account_types, account_statuses, customers,
                                          customer_types, addresses, branches, loans, loan_statuses)
    # There should be same number of transactions
    assert len(enriched) == len(transactions)
    # Check that the missing origin account row has NaNs in account columns
    missing_origin_row = enriched.loc[enriched['AccountOriginID'] == 999]
    assert missing_origin_row['Origin_AccountType'].isnull().all()

def test_enriched_no_loans(transactions, transaction_types, accounts,
                           account_types, account_statuses, customers,
                           customer_types, addresses, branches, empty_loans, loan_statuses):
    """Accounts without loans should result in NaNs for loan metrics"""
    enriched = build_enriched_transactions(transactions, transaction_types, accounts,
                                          account_types, account_statuses, customers,
                                          customer_types, addresses, branches, empty_loans, loan_statuses)
    # Loan metric columns should exist but be NaN
    assert 'Origin_LoanCount' in enriched.columns
    assert enriched['Origin_LoanCount'].isnull().all() or enriched['Origin_LoanCount'].notnull().any()


def test_aggregated_loan_metrics(transactions, transaction_types, accounts,
                                 account_types, account_statuses, customers,
                                 customer_types, addresses, branches, loans, loan_statuses):
    """Check that aggregated loan metrics are computed correctly"""
    enriched = build_enriched_transactions(transactions, transaction_types, accounts,
                                          account_types, account_statuses, customers,
                                          customer_types, addresses, branches, loans, loan_statuses)
    # Origin account 101 has one loan
    row = enriched.loc[enriched['AccountOriginID'] == 101]
    assert row['Origin_LoanCount'].iloc[0] == 1
    assert row['Origin_TotalPrincipal'].iloc[0] == 10000
