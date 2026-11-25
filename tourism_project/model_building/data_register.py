"""
Data Registration
-----------------
Registers the raw dataset that lives inside the GitHub repo
(tourism_project/data/tourism.csv):
  * checks the file exists and is readable,
  * validates that every expected column is present,
  * prints a short summary (shape, dtypes, missing values, target balance),
  * writes a small registry record (data/data_registry.json) with a checksum
    so every pipeline run can be traced back to the exact data version.
The script exits with a non-zero code if validation fails, which stops the
GitHub Actions pipeline before any model is trained on bad data.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "tourism_project/data/tourism.csv")
REGISTRY_PATH = os.path.join(os.path.dirname(DATA_PATH), "data_registry.json")
TARGET = "ProdTaken"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def file_md5(path: str) -> str:
    """Checksum used as the dataset version id."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # 1. File must exist
    if not os.path.exists(DATA_PATH):
        sys.exit(f"[ERROR] Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 2. Schema validation: every expected column must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        sys.exit(f"[ERROR] Missing expected columns: {missing}")
    print("[OK] All expected columns are present.")
    if extra:
        print(f"[INFO] Extra columns found (will be dropped in prep): {extra}")

    # 3. Basic sanity checks on the target
    if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
        sys.exit("[ERROR] Target 'ProdTaken' must be binary (0/1) with no missing values.")
    if len(df) < 100:
        sys.exit("[ERROR] Dataset has too few rows to train a model.")

    # 4. Summary
    print("\n=========== DATASET SUMMARY ===========")
    print(f"File            : {DATA_PATH}")
    print(f"Rows x Columns  : {df.shape[0]} x {df.shape[1]}")
    print(f"Duplicate rows  : {df.duplicated().sum()}")
    print(f"Total missing   : {int(df.isna().sum().sum())}")
    print("\nTarget distribution (ProdTaken):")
    print(df[TARGET].value_counts().rename({0: "0 - Not purchased", 1: "1 - Purchased"}).to_string())
    print(f"Purchase rate   : {df[TARGET].mean():.2%}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("\nCategorical levels:")
    for col in df.select_dtypes(include="object").columns:
        print(f"  {col:15s}: {sorted(df[col].dropna().unique().tolist())}")
    print("\nNumeric summary:")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())

    # 5. Registry record
    record = {
        "dataset": os.path.basename(DATA_PATH),
        "path": DATA_PATH,
        "md5": file_md5(DATA_PATH),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "purchase_rate": round(float(df[TARGET].mean()), 4),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[OK] Dataset registered -> {REGISTRY_PATH}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
