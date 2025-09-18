from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)
import pandas as pd

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data and return a results DataFrame.

    Parameters
    ----------
    models : dict
        Dictionary containing trained models. Keys are model names, values are model objects.
    X_test : pd.DataFrame
        Test features to evaluate the models on.
    y_test : pd.Series
        True labels corresponding to X_test.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame summarizing model performance, including metrics such as ROC-AUC, PR-AUC,
        Precision, Recall, Accuracy, Balanced Accuracy, and optionally confusion matrices
        and classification reports.
    """    
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        results.append({
            "model": name,
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, zero_division=0)
        })

    results_df = pd.DataFrame(results).set_index("model")
    return results_df