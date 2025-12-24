# House Price Prediction (Ames Housing)

## Objective
Predict house sale prices using regression models and a production-style preprocessing pipeline.

## Tech
Python, Pandas, scikit-learn, Matplotlib/Seaborn, joblib

## Model Artifacts
Trained models (`.pkl` files) are not stored in the repository.

## How to run
pip install -r requirements.txt  
python src/01_clean_data.py  
python src/02_build_pipeline.py  
python src/03_train_models.py  
python src/04_evaluate_and_explain.py  

## Outputs
- data/processed/cleaned_house_data.csv
- models/feature_pipeline.pkl
- models/best_model.pkl
- reports/model_results.csv
- reports/figures/pred_vs_actual.png
- reports/results_summary.md
