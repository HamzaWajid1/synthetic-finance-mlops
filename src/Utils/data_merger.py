import pandas as pd



def build_enriched_transactions(transactions, transaction_types,
                                accounts, account_types, account_statuses,
                                customers, customer_types, addresses,
                                branches, loans, loan_statuses):
    """
    Build an enriched transaction dataset by merging multiple financial tables.
    
    Parameters:
    -----------
    transactions : pd.DataFrame
        Core transaction data
    transaction_types : pd.DataFrame
        Transaction type lookup table
    accounts : pd.DataFrame
        Account information
    account_types : pd.DataFrame
        Account type lookup table
    account_statuses : pd.DataFrame
        Account status lookup table
    customers : pd.DataFrame
        Customer information
    customer_types : pd.DataFrame
        Customer type lookup table
    addresses : pd.DataFrame
        Address information
    branches : pd.DataFrame
        Branch information
    loans : pd.DataFrame
        Loan information
    loan_statuses : pd.DataFrame
        Loan status lookup table
    
    Returns:
    --------
    pd.DataFrame
        Enriched transaction dataset with all relevant information
    """
    
    # === STEP 1: Transaction types ===
    tx = transactions.merge(
        transaction_types,
        on="TransactionTypeID",
        how="left"
    ).rename(columns={"TypeName": "TransactionTypeName"})

    # === STEP 2: Origin account details ===
    tx = tx.merge(
        accounts.add_prefix("Origin_"),
        left_on="AccountOriginID",
        right_on="Origin_AccountID",
        how="left"
    )
    tx = tx.merge(
        account_types.add_prefix("Origin_"),
        left_on="Origin_AccountTypeID",
        right_on="Origin_AccountTypeID",
        how="left"
    ).rename(columns={"Origin_TypeName": "Origin_AccountType"})
    tx = tx.merge(
        account_statuses.add_prefix("Origin_"),
        left_on="Origin_AccountStatusID",
        right_on="Origin_AccountStatusID",
        how="left"
    ).rename(columns={"Origin_StatusName": "Origin_AccountStatus"})

    # === STEP 3: Destination account details ===
    tx = tx.merge(
        accounts.add_prefix("Dest_"),
        left_on="AccountDestinationID",
        right_on="Dest_AccountID",
        how="left"
    )
    tx = tx.merge(
        account_types.add_prefix("Dest_"),
        left_on="Dest_AccountTypeID",
        right_on="Dest_AccountTypeID",
        how="left"
    ).rename(columns={"Dest_TypeName": "Dest_AccountType"})
    tx = tx.merge(
        account_statuses.add_prefix("Dest_"),
        left_on="Dest_AccountStatusID",
        right_on="Dest_AccountStatusID",
        how="left"
    ).rename(columns={"Dest_StatusName": "Dest_AccountStatus"})

    # === STEP 4: Customer info (origin & dest) ===
    customers_full = customers.merge(
        customer_types,
        on="CustomerTypeID",
        how="left"
    ).merge(
        addresses,
        on="AddressID",
        how="left"
    )

    # Origin customer
    tx = tx.merge(
        customers_full.add_prefix("Origin_"),
        left_on="Origin_CustomerID",
        right_on="Origin_CustomerID",
        how="left"
    )

    # Destination customer
    tx = tx.merge(
        customers_full.add_prefix("Dest_"),
        left_on="Dest_CustomerID",
        right_on="Dest_CustomerID",
        how="left"
    )

    # === STEP 5: Branch info ===
    branches_full = branches.merge(
        addresses.add_prefix("Branch_"),
        left_on="AddressID",
        right_on="Branch_AddressID",
        how="left"
    )

    tx = tx.merge(
        branches_full.add_prefix("Branch_"),
        left_on="BranchID",
        right_on="Branch_BranchID",
        how="left"
    )

    # === STEP 6: Loan features ===
    # Aggregate loan metrics per account
    loan_features = loans.groupby("AccountID").agg(
        LoanCount=("LoanID", "count"),
        TotalPrincipal=("PrincipalAmount", "sum"),
        AvgInterestRate=("InterestRate", "mean"),
        MaxInterestRate=("InterestRate", "max"),
        MinInterestRate=("InterestRate", "min")
    ).reset_index()

    # Loan status counts per account
    loan_status_pivot = loans.merge(
        loan_statuses,
        on="LoanStatusID",
        how="left"
    ).groupby(["AccountID", "StatusName"]).size().unstack(fill_value=0).reset_index()

    # Merge loan features for origin accounts
    tx = tx.merge(
        loan_features.add_prefix("Origin_"),
        left_on="Origin_AccountID",
        right_on="Origin_AccountID",
        how="left"
    )

    # Merge loan features for destination accounts
    tx = tx.merge(
        loan_features.add_prefix("Dest_"),
        left_on="Dest_AccountID",
        right_on="Dest_AccountID",
        how="left"
    )

    # Merge loan status counts for origin accounts
    tx = tx.merge(
        loan_status_pivot.add_prefix("Origin_LoanStatus_"),
        left_on="Origin_AccountID",
        right_on="Origin_LoanStatus_AccountID",
        how="left"
    )

    # Merge loan status counts for destination accounts
    tx = tx.merge(
        loan_status_pivot.add_prefix("Dest_LoanStatus_"),
        left_on="Dest_AccountID",
        right_on="Dest_LoanStatus_AccountID",
        how="left"
    )

    return tx