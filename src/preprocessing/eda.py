from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "archive (5)" / "fraud.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("FINANCIAL FRAUD DETECTION - DATASET ANALYSIS")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n1. DATASET OVERVIEW")
print("-" * 40)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n2. FIRST FIVE RECORDS")
print("-" * 40)
print(df.head())

print("\n3. DATA TYPES")
print("-" * 40)
print(df.dtypes)

print("\n4. MISSING VALUES")
print("-" * 40)
print(df.isnull().sum())

print("\n5. DUPLICATE RECORDS")
print("-" * 40)
print(df.duplicated().sum())

print("\n6. FRAUD DISTRIBUTION")
print("-" * 40)
print(df["isFraud"].value_counts())

fraud_percentage = df["isFraud"].mean() * 100
print(f"\nFraud percentage: {fraud_percentage:.4f}%")

print("\n7. TRANSACTION TYPES")
print("-" * 40)
print(df["type"].value_counts())

print("\n8. TRANSACTION AMOUNT STATISTICS")
print("-" * 40)
print(df["amount"].describe())

print("\n9. FRAUD BY TRANSACTION TYPE")
print("-" * 40)
print(pd.crosstab(df["type"], df["isFraud"]))

# Save summary
summary = {
    "total_transactions": len(df),
    "fraud_transactions": int(df["isFraud"].sum()),
    "legitimate_transactions": int((df["isFraud"] == 0).sum()),
    "fraud_percentage": fraud_percentage,
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_records": int(df.duplicated().sum()),
}

pd.Series(summary).to_csv(REPORT_DIR / "eda_summary.csv")

print("\nEDA summary saved to reports/eda_summary.csv")
print("\nDATASET ANALYSIS COMPLETED")