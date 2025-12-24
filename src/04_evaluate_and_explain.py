import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/cleaned_house_data.csv")
MODEL_PATH = Path("models/best_model.pkl")
SUMMARY_PATH = Path("reports/results_summary.md")
PLOT_PATH = Path("reports/figures/pred_vs_actual.png")

def main():
    df = pd.read_csv(DATA_PATH)
    target = "SalePrice"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Predicted vs Actual House Prices")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    model_step = model.named_steps["model"]
    preprocess = model.named_steps["preprocess"]

    top_features_text = "Feature importance not available for this model type."

    if hasattr(model_step, "feature_importances_"):
        cat_encoder = preprocess.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = preprocess.transformers_[1][2]
        num_cols = preprocess.transformers_[0][2]
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
        feature_names = list(num_cols) + list(cat_feature_names)

        importances = model_step.feature_importances_
        fi = pd.Series(importances, index = feature_names).sort_values(ascending=False).head(15)

        top_features_text = "\n".join([f"- {idx}: {val:.4f}" for idx, val in fi.items()])
    
    summary = f"""# House Price Prediction — Results Summary

## What this project does
This model predicts **house sale price** from property characteristics (size, quality, location, etc.).
It demonstrates a production-style ML workflow: cleaning → EDA → pipeline preprocessing → model comparison → evaluation.

## Model evaluation
- Best model saved at: `models/best_model.pkl`
- Predicted vs actual plot: `{PLOT_PATH}`

## Key drivers of price (top signals)
{top_features_text}

## Business interpretation (example)
- **Quality and size** features typically increase price the most (better build quality, larger living area).
- **Location** (e.g., neighborhood) can create large price differences even for similar houses.
- The model can support:
  - pricing strategy for sellers/agents
  - valuation checks for mortgage/underwriting
  - identifying undervalued properties for investment
"""

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print(f"Saved plot -> {PLOT_PATH}")
    print(f"Saved summary -> {SUMMARY_PATH}")

if __name__ == "__main__":
    main()