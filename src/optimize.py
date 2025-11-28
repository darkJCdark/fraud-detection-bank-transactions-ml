from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def optimize_random_forest(X_train, y_train, cv: int = 3, scoring: str = "recall"):
    """
    Optimize Random Forest hyperparameters using GridSearchCV.
    Args:
        X_train: Training features
        y_train: Training labels
        cv (int): Number of cross-validation folds
        scoring (str): Metric to optimize (default: recall)
    Returns:
        dict: Best parameters found
    """
    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, 30, None],
        "max_features": [None, "sqrt", "log2"]
    }

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)

    # GridSearchCV
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)
    print("Best recall score:", grid.best_score_)

    return grid.best_params_
