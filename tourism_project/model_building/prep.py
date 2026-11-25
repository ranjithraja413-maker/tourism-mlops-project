"""
Data Preparation
----------------
  * loads the dataset directly from the repository data folder,
  * cleans it (drops index/ID columns, fixes inconsistent labels,
    merges Unmarried into Single, ordinal-encodes ProductPitched and Designation,
    removes duplicates and rows with a missing target),
  * splits it into stratified train/test sets,
  * saves Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally
    (GitHub Actions then uploads them as a workflow artifact for the
    training job).
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.getenv("DATA_PATH", "tourism_project/data/tourism.csv")
OUTPUT_DIR = os.getenv("SPLIT_DIR", ".")   # repo root -> matches pipeline.yml artifact paths
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ordered categories -> integer codes (the same maps are used in the Streamlit app)
ORDINAL_MAPS = {
    "ProductPitched": {"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3, "King": 4},
    "Designation": {"Executive": 0, "Manager": 1, "Senior Manager": 2, "AVP": 3, "VP": 4},
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove unnecessary columns: the leftover pandas index and the customer ID
    #    carry no predictive information (pure identifiers).
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"Dropped columns: {drop_cols}")

    # Strip stray whitespace in every text column
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Fix inconsistent category labels
    # Gender: 'Fe Male' is a typo of 'Female'
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    print("Gender levels        :", df["Gender"].value_counts().to_dict())
    # Occupation: 'Free Lancer' -> 'Freelancer' (spelling of the data dictionary)
    df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})
    # MaritalStatus: 'Unmarried' and 'Single' describe the same group -> combine
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    print("MaritalStatus levels :", df["MaritalStatus"].value_counts().to_dict())

    # 3. Ordinal encoding: these categories have a natural order
    #    ProductPitched -> package tier (cheapest to most premium)
    #    Designation    -> seniority in the organisation
    for col, mapping in ORDINAL_MAPS.items():
        unknown = set(df[col].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected {col} values: {unknown}")
        df[col] = df[col].map(mapping).astype(int)
        print(f"{col} encoded        : {mapping}")

    # 4. Drop rows without a target and exact duplicates
    before = len(df)
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    print(f"Removed {before - len(df)} duplicate / unlabeled rows")

    # 5. Impute any remaining missing values (none in the current file, but keeps
    #    the pipeline robust when new data arrives)
    for col in df.columns:
        if df[col].isna().any():
            fill = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
            df[col] = df[col].fillna(fill)
            print(f"Imputed missing values in {col}")

    df[TARGET] = df[TARGET].astype(int)
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape}")

    df = clean(df)
    print(f"Cleaned dataset shape: {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split keeps the ~19% purchase rate identical in train and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Xtrain.to_csv(os.path.join(OUTPUT_DIR, "Xtrain.csv"), index=False)
    Xtest.to_csv(os.path.join(OUTPUT_DIR, "Xtest.csv"), index=False)
    ytrain.to_csv(os.path.join(OUTPUT_DIR, "ytrain.csv"), index=False)
    ytest.to_csv(os.path.join(OUTPUT_DIR, "ytest.csv"), index=False)

    print(f"Train: {Xtrain.shape}, purchase rate {ytrain.mean():.2%}")
    print(f"Test : {Xtest.shape}, purchase rate {ytest.mean():.2%}")
    print(f"Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
