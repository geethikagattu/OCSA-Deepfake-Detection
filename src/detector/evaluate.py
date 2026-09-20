import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2
            else None
        ),
    }
