"""
Evaluation Metrics for Kaggle Competitions.
Specialized for ROC-AUC scoring, F1-macro, Log Loss, and Multi-Class support.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss


def compute_roc_auc(
    y_true: Union[np.ndarray, list],
    y_pred_probs: Union[np.ndarray, list],
    multi_class: str = "ovr",
    average: str = "macro",
) -> float:
    """
    Computes Area Under the ROC Curve (ROC-AUC).
    Gracefully handles both 1D and 2D probability matrices.

    Args:
        y_true: Ground truth binary or multiclass labels (shape: [N]).
        y_pred_probs: Predicted class probabilities (shape: [N] or [N, num_classes]).
        multi_class: 'ovr' (One-vs-Rest) or 'ovo' (One-vs-One).
        average: 'macro' or 'weighted'.

    Returns:
        ROC-AUC float score (0.0 to 1.0). Returns 0.5 if only one class exists in y_true.
    """
    y_true_arr = np.asarray(y_true)
    probs_arr = np.asarray(y_pred_probs)

    # Edge case: if only 1 unique class is present in the true labels
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        return 0.5

    # Binary classification case
    if len(unique_classes) == 2:
        # If probabilities are 2D ([N, 2]), extract positive class column (index 1)
        if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
            probs_arr = probs_arr[:, 1]
        elif probs_arr.ndim == 2 and probs_arr.shape[1] == 1:
            probs_arr = probs_arr.squeeze(1)

        try:
            return float(roc_auc_score(y_true_arr, probs_arr))
        except ValueError:
            return 0.5

    # Multi-class case
    try:
        return float(roc_auc_score(y_true_arr, probs_arr, multi_class=multi_class, average=average))
    except ValueError:
        return 0.5


def compute_eval_metrics(
    y_true: Union[np.ndarray, list],
    y_pred_probs: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculates a comprehensive evaluation dictionary:
    ROC-AUC, Macro-F1, Accuracy, and Log Loss.
    """
    y_true_arr = np.asarray(y_true)
    probs_arr = np.asarray(y_pred_probs)

    auc = compute_roc_auc(y_true_arr, probs_arr)

    # Determine discrete predictions
    if probs_arr.ndim == 1:
        preds = (probs_arr >= threshold).astype(int)
    elif probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
        preds = (probs_arr[:, 1] >= threshold).astype(int)
    else:
        preds = np.argmax(probs_arr, axis=1)

    acc = float(accuracy_score(y_true_arr, preds))
    f1 = float(f1_score(y_true_arr, preds, average="macro", zero_division=0))

    metrics = {
        "roc_auc": round(auc, 5),
        "f1_macro": round(f1, 5),
        "accuracy": round(acc, 5),
    }

    try:
        loss = float(log_loss(y_true_arr, probs_arr))
        metrics["log_loss"] = round(loss, 5)
    except Exception:
        pass

    return metrics
