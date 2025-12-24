import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RAW_CLEAN_PATH = Path("data/processed/cleaned_house_data.csv")
PIPELINE_PATH = Path("models/feauture_pipeline.pkl")

def main():

    df = pd.read_csv(RAW_CLEAN_PATH)
    target = "SalePrice"

    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    #numeric pipeline : impute -> scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical Pipeline : impute -> one -hot
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine into single Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Save only the feature pipeline (preprocessor)
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PIPELINE_PATH)

    # also do split herre to confirm no errors
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor.fit(X_train)

    print(f"Saved feature pipeline -> {PIPELINE_PATH}")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

if __name__ == "__main__":
    main()