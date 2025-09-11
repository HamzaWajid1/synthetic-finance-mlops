# Synthetic Finance Anomaly Detection Project

This project demonstrates how to build an **enriched transaction dataset** for anomaly detection in personal finance. It merges multiple synthetic financial tables into a single transaction-level DataFrame and computes **initial features** that are useful for detecting suspicious or unusual transactions.

---



---

## 📄 Dataset Description

The synthetic dataset contains the following tables:

- **transactions** (50,000 rows)  
- **transaction_types** (Deposit, Withdrawal, Transfer, etc.)  
- **accounts** (AccountID, CustomerID, Balance, etc.)  
- **account_types** (Checking, Savings, Payroll, etc.)  
- **account_statuses** (Active, Inactive, Closed)  
- **customers** (CustomerID, Name, DateOfBirth, CustomerTypeID, AddressID)  
- **customer_types** (Individual, Small Business, Enterprise)  
- **addresses** (Street, City, Country)  
- **branches** (BranchID, BranchName, AddressID)  
- **loans** (LoanID, AccountID, PrincipalAmount, InterestRate, LoanStatusID)  
- **loan_statuses** (Active, Paid Off, Overdue)

---

## 🔹 Step 1: Merge Tables

The goal is to produce a **single enriched transaction DataFrame**, containing all relevant information about origin and destination accounts, customers, loans, and branches.

1. **Prepare Accounts Table**
    ```python
    accounts_full = accounts.merge(account_types, on="AccountTypeID", how="left") \
                            .merge(account_statuses, on="AccountStatusID", how="left")
    ```
    - Combines account balance, type, and status.

2. **Prepare Customers Table**
    ```python
    customers_full = customers.merge(customer_types, on="CustomerTypeID", how="left") \
                              .merge(addresses, on="AddressID", how="left")
    ```
    - Adds customer type and address information.

3. **Prepare Loan Features**
    ```python
    loan_features = loans.groupby("AccountID").agg(
        LoanCount=("LoanID", "count"),
        TotalPrincipal=("PrincipalAmount", "sum"),
        AvgInterestRate=("InterestRate", "mean")
    ).reset_index()
    ```
    - Aggregates numeric loan metrics per account.
    
    ```python
    loan_status_pivot = loans.merge(loan_statuses, on="LoanStatusID", how="left") \
                             .groupby(["AccountID", "StatusName"]).size() \
                             .unstack(fill_value=0).reset_index()
    ```
    - Counts number of loans per loan status.

4. **Merge Transactions with Transaction Types**
    ```python
    tx = transactions.merge(transaction_types, on="TransactionTypeID", how="left")
    tx.rename(columns={"TypeName": "TransactionTypeName"}, inplace=True)
    ```

5. **Merge Origin & Destination Accounts**
    ```python
    # Origin
    tx = tx.merge(accounts_full.add_prefix("Origin_"), left_on="AccountOriginID", right_on="Origin_AccountID", how="left")
    
    # Destination
    tx = tx.merge(accounts_full.add_prefix("Dest_"), left_on="AccountDestinationID", right_on="Dest_AccountID", how="left")
    ```

6. **Merge Origin & Destination Customers**
    ```python
    # Origin
    tx = tx.merge(customers_full.add_prefix("Origin_"), left_on="Origin_CustomerID", right_on="Origin_CustomerID", how="left")
    
    # Destination
    tx = tx.merge(customers_full.add_prefix("Dest_"), left_on="Dest_CustomerID", right_on="Dest_CustomerID", how="left")
    ```

7. **Merge Loan Features for Origin & Destination**
    ```python
    tx = tx.merge(loan_features.add_prefix("Origin_"), left_on="Origin_AccountID", right_on="Origin_AccountID", how="left")
    tx = tx.merge(loan_features.add_prefix("Dest_"), left_on="Dest_AccountID", right_on="Dest_AccountID", how="left")
    ```

8. **Merge Loan Status Counts**
    ```python
    tx = tx.merge(loan_status_pivot.add_prefix("Origin_LoanStatus_"), left_on="Origin_AccountID", right_on="Origin_LoanStatus_AccountID", how="left")
    tx = tx.merge(loan_status_pivot.add_prefix("Dest_LoanStatus_"), left_on="Dest_AccountID", right_on="Dest_LoanStatus_AccountID", how="left")
    ```

9. **Optionally Merge Branch Information**
    ```python
    tx = tx.merge(branches.add_prefix("Branch_"), left_on="BranchID", right_on="Branch_BranchID", how="left")
    tx = tx.merge(addresses.add_prefix("Branch_Addr_"), left_on="Branch_AddressID", right_on="Branch_Addr_AddressID", how="left")
    ```

> At the end of these steps, `tx` is a single enriched DataFrame with **all relevant columns** from origin/destination accounts, customers, loans, and branches.

---

## 🔹 Step 2: Add Transaction Features

After merging, we compute **initial anomaly features**:

```python
# Amount ratios
tx['Amount_to_OriginBalance'] = tx['Amount'] / tx['Origin_Balance'].replace(0, np.nan)
tx['Amount_to_DestBalance'] = tx['Amount'] / tx['Dest_Balance'].replace(0, np.nan)

# Account flags
tx['Origin_AccountInactive'] = tx['Origin_AccountStatus'].isin(['Inactive','Closed']).astype(int)
tx['Dest_AccountInactive'] = tx['Dest_AccountStatus'].isin(['Inactive','Closed']).astype(int)

# Customer age
today = pd.Timestamp.today()
tx['Origin_Age'] = (today - pd.to_datetime(tx['Origin_DateOfBirth'], errors='coerce')).dt.days // 365
tx['Dest_Age'] = (today - pd.to_datetime(tx['Dest_DateOfBirth'], errors='coerce')).dt.days // 365

# Loan leverage ratios
tx['Origin_LoanLeverage'] = tx['Origin_TotalPrincipal'] / tx['Origin_Balance'].replace(0, np.nan)
tx['Dest_LoanLeverage'] = tx['Dest_TotalPrincipal'] / tx['Dest_Balance'].replace(0, np.nan)

# Time features
tx['TransactionHour'] = pd.to_datetime(tx['TransactionDate']).dt.hour
tx['TransactionWeekday'] = pd.to_datetime(tx['TransactionDate']).dt.dayofweek

# Flags for anomaly heuristics
tx['LargeTransferFlag'] = (tx['Amount_to_OriginBalance'] > 0.5).astype(int)
```

### ✅ Feature Categories

| Category             | Example Features                                                        |
| -------------------- | ----------------------------------------------------------------------- |
| Transaction Amount   | Amount, Amount_to_OriginBalance, Amount_to_DestBalance              |
| Account Info         | Balance, AccountType, AccountStatus, AccountInactive flag               |
| Customer Info        | Age, CustomerType                                                       |
| Loan Info            | LoanCount, TotalPrincipal, AvgInterestRate, LoanLeverage, Overdue loans |
| Transaction Patterns | Hour of day, Weekday, LargeTransferFlag                                 |

---

## 🔹 Next Steps

1. **Feature Engineering**: Standardize numeric features, encode categorical features.
2. **Anomaly Detection Modeling**: Use Isolation Forest, Autoencoder, or XGBoost to detect anomalies.
3. **Visualization & Dashboards**: Plot distributions, anomalies, and statistics using Streamlit or Plotly.
4. **MLOps & Deployment**: Track experiments with MLflow, deploy API with FastAPI + Docker.

---

## References

* Synthetic Finance Dataset (Kaggle / TestDataBox)
* Pandas `merge()` documentation: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)
