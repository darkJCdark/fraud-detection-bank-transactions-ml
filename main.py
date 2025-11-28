# main.py
"""
Main script to run the full fraud detection pipeline.
Steps:
1. Preprocess data
2. Feature engineering
3. Train models
4. Optimize Random Forest
5. Evaluate models
"""

from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.train import train_models
from src.evaluate import evaluate_models
from src.optimize import optimize_random_forest

def main():
    # 1. Load and preprocess dataset
    file_path = "data/fraud_data.csv"
    print("Loading and preprocessing data...")
    df = preprocess_pipeline(file_path)

    # 2. Feature engineering
    print("Applying feature engineering...")
    df = feature_engineering_pipeline(df)

    # 3. Train models
    print("Training models...")
    X_train, X_test, y_train, y_test, models = train_models(df)

    # 4. Optimize Random Forest
    print("Optimizing Random Forest hyperparameters...")
    best_params = optimize_random_forest(X_train, y_train)
    print("Best parameters:", best_params)

    # 5. Evaluate models
    print("Evaluating models...")
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
