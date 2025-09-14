# Synthetic Finance Dataset — Customers, Accounts, Loans & Transactions (50k+ rows)

This dataset contains **synthetic financial data** designed for practicing **fraud detection, credit risk modeling, data cleaning, and SQL joins**.  
It simulates a mini financial system with customers, accounts, loans, transactions, and reference tables.

The dataset is **100% synthetic** — no personal or sensitive data is included.  
Errors and anomalies were introduced intentionally to reflect real-world data quality issues.

---

## 🗂️ Tables Included

### Core Entities
- **customers.csv** → Customer master data (id, type, demographics).  
- **accounts.csv** → Bank accounts linked to customers, with type & status.  
- **loans.csv** → Loans issued, with type, status, and customer references.  
- **transactions.csv** → Financial transactions with customer, account, branch, and type references.  

### Dimensions & Reference Tables
- **customer_types.csv** → Classification of customers (e.g. personal, corporate).  
- **account_types.csv** → Account categories (e.g. savings, checking).  
- **account_statuses.csv** → Account lifecycle states (e.g. active, closed).  
- **loan_statuses.csv** → Loan states (e.g. current, default).  
- **transaction_types.csv** → Transaction categories (e.g. payment, transfer, withdrawal).  
- **branches.csv** → Bank branch metadata (id, name, location).  
- **addresses.csv** → Address data linked to customers and branches.  

---

## 🔗 Relationships (ERD overview)

- `customers` ↔ `accounts` (1:N)  
- `customers` ↔ `loans` (1:N)  
- `accounts` ↔ `transactions` (1:N)  
- `branches` ↔ `transactions` (1:N)  
- Reference tables provide valid values for types/statuses.  

👉 This makes it possible to practice **joins, constraints, and integrity checks**.

---

## ⚠️ Data Quality Issues (intentionally added)

To simulate real-world scenarios, ~7% of the data includes errors:  
- 2% missing values (nulls)  
- 1% typos in text fields  
- 1% duplicate rows  
- 1% inconsistent number/text formats  
- 1% mixed date formats  
- 1% non-sequential IDs  
- 1% future dates  

These anomalies are useful for **ETL testing, anomaly detection, and data validation**.

---

## 🎯 Example Use Cases
- Fraud detection models with transaction-level data  
- Credit risk scoring with loans & defaults  
- Data cleaning pipelines (handling nulls, duplicates, inconsistencies)  
- SQL practice with multiple related tables  
- Teaching data governance and data quality concepts  

---

## 📊 Example Fields
- `transaction_id`, `account_id`, `customer_id`, `branch_id`, `transaction_type_id`, `amount`, `transaction_datetime`, `is_fraud`  
- `loan_id`, `loan_status_id`, `loan_amount`, `interest_rate`  
- `account_id`, `account_type_id`, `account_status_id`, `balance`  
- `customer_id`, `customer_type_id`, `address_id`, `date_of_birth`  

---

## 🔄 Reproducibility
This dataset is **synthetic and reproducible**.  
You can generate larger or smaller datasets, adjust error percentages, and export to formats like CSV, Parquet or Avro.

👉 Generated with [TestDataBox](https://testdatabox.com/?utm_source=kaggle)  

---

## 📜 License
- Data: CC BY 4.0  
- Code & Notebooks: MIT  

---

## 📌 Changelog
- **v1.0.0** — Initial release (11 CSV files, ~50k rows, ~7% noise).
