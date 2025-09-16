import numpy as np
import pandas as pd



def compute_transaction_features(df):
    """
    Compute derived features for anomaly detection from the enriched transaction dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Enriched transaction dataset with merged account, customer, and loan information
    
    Returns:
    --------
    pd.DataFrame
        Dataset with additional engineered features
    """
    
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    print("🔧 Computing transaction features...")
    
    # === AMOUNT-BASED FEATURES ===
    print("  → Amount-based features")
    
    # Amount ratios (avoid division by zero)
    df['Amount_to_OriginBalance'] = df['Amount'] / df['Origin_Balance'].replace(0, np.nan)
    df['Amount_to_DestBalance'] = df['Amount'] / df['Dest_Balance'].replace(0, np.nan)
    
    # Amount relative to average transaction size
    avg_transaction = df['Amount'].mean()
    df['Amount_to_AvgTransaction'] = df['Amount'] / avg_transaction
    
    # === ACCOUNT STATUS FEATURES ===
    print("  → Account status features")
    
    # Account inactivity flags
    df['Origin_AccountInactive'] = df['Origin_AccountStatus'].isin(['Inactive', 'Closed']).astype(int)
    df['Dest_AccountInactive'] = df['Dest_AccountStatus'].isin(['Inactive', 'Closed']).astype(int)
    
    # Account type flags
    df['Origin_IsChecking'] = (df['Origin_AccountType'] == 'Checking').astype(int)
    df['Dest_IsChecking'] = (df['Dest_AccountType'] == 'Checking').astype(int)
    df['Origin_IsSavings'] = (df['Origin_AccountType'] == 'Savings').astype(int)
    df['Dest_IsSavings'] = (df['Dest_AccountType'] == 'Savings').astype(int)
    
    # === CUSTOMER DEMOGRAPHIC FEATURES ===
    print("  → Customer demographic features")
    
    # Calculate customer ages
    today = pd.Timestamp.today()
    df['Origin_Age'] = (today - pd.to_datetime(df['Origin_DateOfBirth'], errors='coerce')).dt.days // 365
    df['Dest_Age'] = (today - pd.to_datetime(df['Dest_DateOfBirth'], errors='coerce')).dt.days // 365
    
    # Age difference between origin and destination customers
    df['Age_Difference'] = df['Origin_Age'] - df['Dest_Age']
    
    # Customer type flags
    df['Origin_IsIndividual'] = (df['Origin_TypeName'] == 'Individual').astype(int)
    df['Dest_IsIndividual'] = (df['Dest_TypeName'] == 'Individual').astype(int)
    df['Origin_IsBusiness'] = (df['Origin_TypeName'] == 'Small Business').astype(int)
    df['Dest_IsBusiness'] = (df['Dest_TypeName'] == 'Small Business').astype(int)
    
    # === LOAN-RELATED FEATURES ===
    print("  → Loan-related features")
    
    # Loan leverage ratios (debt to balance)
    df['Origin_LoanLeverage'] = df['Origin_TotalPrincipal'] / df['Origin_Balance'].replace(0, np.nan)
    df['Dest_LoanLeverage'] = df['Dest_TotalPrincipal'] / df['Dest_Balance'].replace(0, np.nan)
    
    # Loan count flags
    df['Origin_HasLoans'] = (df['Origin_LoanCount'] > 0).astype(int)
    df['Dest_HasLoans'] = (df['Dest_LoanCount'] > 0).astype(int)
    
    # High interest rate flags (above 10%)
    df['Origin_HighInterest'] = (df['Origin_AvgInterestRate'] > 10).astype(int)
    df['Dest_HighInterest'] = (df['Dest_AvgInterestRate'] > 10).astype(int)
    
    # === TEMPORAL FEATURES ===
    print("  → Temporal features")
    
    # Extract time components
    df['TransactionHour'] = pd.to_datetime(df['TransactionDate']).dt.hour
    df['TransactionWeekday'] = pd.to_datetime(df['TransactionDate']).dt.dayofweek
    df['TransactionMonth'] = pd.to_datetime(df['TransactionDate']).dt.month
    df['TransactionQuarter'] = pd.to_datetime(df['TransactionDate']).dt.quarter
    
    # Time-based flags
    df['IsWeekend'] = (df['TransactionWeekday'] >= 5).astype(int)
    df['IsBusinessHours'] = ((df['TransactionHour'] >= 9) & (df['TransactionHour'] <= 17)).astype(int)
    df['IsNightTime'] = ((df['TransactionHour'] >= 22) | (df['TransactionHour'] <= 6)).astype(int)
    
    # === ANOMALY FLAGS ===
    print("  → Anomaly detection flags")
    
    # Large transfer flags
    df['LargeTransferFlag'] = (df['Amount_to_OriginBalance'] > 0.5).astype(int)
    df['VeryLargeTransferFlag'] = (df['Amount_to_OriginBalance'] > 0.9).astype(int)
    
    # Unusual timing flags
    df['UnusualTimingFlag'] = ((df['IsNightTime'] == 1) | (df['IsWeekend'] == 1)).astype(int)
    
    # High-risk account combinations
    df['HighRiskFlag'] = ((df['Origin_AccountInactive'] == 1) | 
                         (df['Dest_AccountInactive'] == 1) |
                         (df['Origin_LoanLeverage'] > 2) |
                         (df['Dest_LoanLeverage'] > 2)).astype(int)
    
    # Cross-customer type transfers (business to individual, etc.)
    df['CrossTypeTransfer'] = (df['Origin_TypeName'] != df['Dest_TypeName']).astype(int)

    features_to_keep = [
    "TransactionTypeID", "Amount",
    "Origin_AccountTypeID", "Origin_AccountStatusID", "Origin_Balance",
    "Dest_AccountTypeID", "Dest_AccountStatusID", "Dest_Balance",
    "Origin_CustomerTypeID", "Dest_CustomerTypeID",
    "Origin_LoanCount", "Origin_TotalPrincipal", "Origin_AvgInterestRate",
    "Dest_LoanCount", "Dest_TotalPrincipal", "Dest_AvgInterestRate",
    "Origin_LoanStatus_Active", "Origin_LoanStatus_Overdue", "Origin_LoanStatus_Paid Off",
    "Dest_LoanStatus_Active", "Dest_LoanStatus_Overdue", "Dest_LoanStatus_Paid Off",
    "Amount_to_OriginBalance", "Amount_to_DestBalance", "Amount_to_AvgTransaction",
    "Origin_AccountInactive", "Dest_AccountInactive", "Age_Difference",
    "Origin_LoanLeverage", "Dest_LoanLeverage",
    "TransactionHour", "TransactionWeekday", "TransactionMonth", "TransactionQuarter",
    "IsWeekend", "IsBusinessHours", "IsNightTime",
    "LargeTransferFlag", "VeryLargeTransferFlag", "UnusualTimingFlag", "HighRiskFlag", "CrossTypeTransfer"
    ]

    df=df[features_to_keep]

    df = df.fillna(0)
    print("✅ Feature engineering completed!")
    
    return df