import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels: 0 = real, 1 = fake.

    y_prob : array-like
        Model probability for fake.

    threshold : float
        Probability threshold used to create predictions.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC requires both classes to be present.
    if len(np.unique(y_true)) == 2:
        results["auc"] = roc_auc_score(y_true, y_prob)
    else:
        results["auc"] = None

    return results
