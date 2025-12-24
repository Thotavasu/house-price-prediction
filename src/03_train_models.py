import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

PIPELINE_PATH = Path("models/feauture_pipeline.pkl")
DATA_PATH = Path("data/processed/cleaned_house_data.csv")
RESULTS_PATH = Path("reports/model_results.csv")

def eval_model(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def main():
    df = pd.read_csv(DATA_PATH)
    target = "SalePrice"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = joblib.load(PIPELINE_PATH)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42, max_iter=20000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
    }

    rows = []
    trained_models = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse, mae, r2 = eval_model(y_test, preds)
        rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        trained_models[name] = pipe

        print(f"{name}: RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.4f}")

    results = pd.DataFrame(rows).sort_values(by="rmse")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    best_name = results.iloc[0]["model"]
    joblib.dump(trained_models[best_name], Path("models/best_model.pkl"))
    print(f"Saved results -> {RESULTS_PATH}")
    print(f"Saved best model -> models/best_model.pkl (best: {best_name})")

if __name__ == "__main__":
    main()