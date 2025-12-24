# House Price Prediction — Results Summary

## What this project does
This model predicts **house sale price** from property characteristics (size, quality, location, etc.).
It demonstrates a production-style ML workflow: cleaning → EDA → pipeline preprocessing → model comparison → evaluation.

## Model evaluation
- Best model saved at: `models/best_model.pkl`
- Predicted vs actual plot: `reports\figures\pred_vs_actual.png`

## Key drivers of price (top signals)
- Overall Qual: 0.6035
- Gr Liv Area: 0.0986
- 1st Flr SF: 0.0383
- Total Bsmt SF: 0.0250
- BsmtFin SF 1: 0.0227
- 2nd Flr SF: 0.0217
- Full Bath: 0.0177
- Garage Cars: 0.0171
- Lot Area: 0.0153
- Garage Area: 0.0147
- Year Built: 0.0090
- PID: 0.0081
- Year Remod/Add: 0.0073
- Bsmt Unf SF: 0.0044
- Mas Vnr Area: 0.0043

## Business interpretation (example)
- **Quality and size** features typically increase price the most (better build quality, larger living area).
- **Location** (e.g., neighborhood) can create large price differences even for similar houses.
- The model can support:
  - pricing strategy for sellers/agents
  - valuation checks for mortgage/underwriting
  - identifying undervalued properties for investment
