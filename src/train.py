import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Split dataset into train and test sets.
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        test_size (float): Proportion of test set.
        random_state (int): Seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(X_train, y_train, max_depth=20, n_estimators=100, max_features=None, random_state=42):
    """
    Train Random Forest model with default or tuned hyperparameters.
    """
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf


def train_models(df: pd.DataFrame):
    """
    Full training pipeline:
    - Split data
    - Train Logistic Regression
    - Train Random Forest
    Returns:
        X_train, X_test, y_train, y_test, models (dict)
    """
    X_train, X_test, y_train, y_test = split_data(df)

    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model
    }

    return X_train, X_test, y_train, y_test, models
