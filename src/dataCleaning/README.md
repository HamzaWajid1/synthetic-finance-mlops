# Data Cleaning Module

This module contains specialized functions for cleaning financial datasets. Each function handles specific data quality issues commonly found in financial data.

## 📁 Module Structure

```
src/dataCleaning/
├── __init__.py              # Module initialization and exports
├── README.md               # This documentation
├── customers_cleaner.py    # Customer data cleaning
├── transactions_cleaner.py # Transaction data cleaning
├── accounts_cleaner.py     # Account data cleaning
├── loans_cleaner.py        # Loan data cleaning
├── addresses_cleaner.py    # Address data cleaning
└── branches_cleaner.py     # Branch data cleaning
```

## 🧹 Cleaning Functions

### 1. **Customer Data Cleaning** (`clean_customers`)
**File:** `customers_cleaner.py`

**What it does:**
- Handles missing values in names and addresses
- Removes duplicate customer records
- Validates customer type IDs (1, 2, 3)
- Standardizes date formats for birth dates
- Removes future birth dates
- Ensures unique customer IDs

**Input:** Raw customer data with potential quality issues
**Output:** Clean, validated customer records

### 2. **Transaction Data Cleaning** (`clean_transactions`)
**File:** `transactions_cleaner.py`

**What it does:**
- Converts amounts to numeric format
- Normalizes amount notation (handles "k" suffix, commas)
- Validates transaction type IDs (1, 2, 3, 4)
- Standardizes transaction dates
- Removes duplicate transactions
- Drops rows with missing critical data (IDs, amounts)
- Removes future-dated transactions

**Input:** Raw transaction data with formatting issues
**Output:** Clean, standardized transaction records

### 3. **Account Data Cleaning** (`clean_accounts`)
**File:** `accounts_cleaner.py`

**What it does:**
- Validates account type IDs (1, 2, 3, 4, 5)
- Validates account status IDs (1, 2, 3)
- Standardizes opening dates
- Ensures numeric format for IDs and balances
- Removes duplicate accounts
- Handles missing opening dates with default values
- Removes future-dated account openings

**Input:** Raw account data with validation issues
**Output:** Clean, validated account records

### 4. **Loan Data Cleaning** (`clean_loans`)
**File:** `loans_cleaner.py`

**What it does:**
- Validates loan status IDs (1, 2, 3)
- Ensures numeric format for amounts and rates
- Standardizes start and end dates
- Removes rows with missing critical data
- Removes duplicate loans
- Removes future-dated loan starts
- Validates principal amounts and interest rates

**Input:** Raw loan data with missing values
**Output:** Clean, complete loan records

### 5. **Address Data Cleaning** (`clean_addresses`)
**File:** `addresses_cleaner.py`

**What it does:**
- Standardizes country names using fuzzy matching
- Formats street and city names (Title Case)
- Handles missing address components
- Ensures unique address IDs
- Validates address ID format
- Removes duplicate addresses

**Input:** Raw address data with formatting inconsistencies
**Output:** Clean, standardized address records

### 6. **Branch Data Cleaning** (`clean_branches`)
**File:** `branches_cleaner.py`

**What it does:**
- Standardizes branch names (Title Case, trimmed)
- Validates branch and address IDs
- Removes duplicate branches
- Ensures numeric format for IDs
- Handles missing branch names

**Input:** Raw branch data with formatting issues
**Output:** Clean, standardized branch records

## 🚀 Usage

### Import Individual Functions
```python
from src.dataCleaning import clean_customers, clean_transactions

# Clean customer data
df_customers = pd.read_csv("data/raw/customers.csv")
cleaned_customers = clean_customers(df_customers)

# Clean transaction data
df_transactions = pd.read_csv("data/raw/transactions.csv")
cleaned_transactions = clean_transactions(df_transactions)
```

### Import All Functions
```python
from src.dataCleaning import *

# Use any cleaning function
cleaned_accounts = clean_accounts(df_accounts)
cleaned_loans = clean_loans(df_loans)
```

### Run Individual Cleaners
```python
# Run as standalone scripts
python src/dataCleaning/customers_cleaner.py
python src/dataCleaning/transactions_cleaner.py
```

## 🔧 Common Data Quality Issues Handled

1. **Missing Values**
   - Fills missing names with "Unknown"
   - Handles missing dates with defaults
   - Drops rows with critical missing data

2. **Data Type Issues**
   - Converts strings to numeric where needed
   - Standardizes date formats
   - Ensures proper ID formats

3. **Duplicate Records**
   - Removes exact duplicates
   - Handles duplicate IDs (keeps first occurrence)

4. **Format Inconsistencies**
   - Normalizes text formatting (Title Case)
   - Handles amount notation (commas, "k" suffix)
   - Standardizes country names

5. **Data Validation**
   - Validates ID ranges against expected values
   - Removes future-dated records
   - Ensures referential integrity

6. **Fuzzy Matching**
   - Uses RapidFuzz for country name standardization
   - Handles typos in categorical fields

## 📊 Data Quality Metrics

Each cleaning function provides:
- **Input shape** - Original number of rows/columns
- **Output shape** - Cleaned number of rows/columns
- **Records removed** - Shows data quality impact

## 🛠️ Dependencies

- `pandas` - Data manipulation
- `rapidfuzz` - Fuzzy string matching
- `numpy` - Numerical operations

## 📝 Notes

- All functions preserve the original data structure
- Functions are designed to be idempotent (safe to run multiple times)
- Error handling uses pandas' `errors="coerce"` for graceful failures
- Date validation prevents future-dated records
- ID validation ensures data integrity

## 🔄 Integration with MLOps Pipeline

These cleaning functions are designed to integrate seamlessly with:
- **Data preprocessing pipelines**
- **Feature engineering workflows**
- **Model training pipelines**
- **Data validation processes**

The cleaned data is ready for the next stages of your MLOps pipeline, including feature engineering, model training, and deployment.
