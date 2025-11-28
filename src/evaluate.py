import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a single model and print metrics.
    Args:
        model: Trained model (Logistic Regression, Random Forest, etc.)
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    """
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {model_name} Evaluation ===")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraud", "Fraud"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluate multiple models stored in a dictionary.
    Args:
        models (dict): Dictionary with model name as key and trained model as value
        X_test: Test features
        y_test: Test labels
    """
    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)
