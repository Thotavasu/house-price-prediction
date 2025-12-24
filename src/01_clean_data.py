import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/AmesHousing.csv")
OUT_PATH = Path("data/processed/cleaned_house_data.csv")

def main():
    df = pd.read_csv(RAW_PATH)

    if "SalePrice" not in df.columns:
        raise ValueError("Target column 'SalesPrice' not found. Check dataset")

    # Identify Column Types
    target = "SalePrice"
    X = df.drop(columns=[target])
    y = df[target]

    # select categorical vs numerical by type
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Handle missing values
    # Numerical columns : fill with median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    #Categorical columns : fill missing with unknown
    df[cat_cols] = df[cat_cols].fillna("unknown")

    # save cleaned dataset
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Print a quick summary for confidence
    missing_after = df.isna().sum().sum()
    print(f"Saved cleaned datset -> {OUT_PATH}")
    print(f"Total missing values after missing -> {missing_after}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

if __name__ == "__main__":
    main()