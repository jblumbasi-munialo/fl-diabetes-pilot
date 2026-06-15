"""
Fairness evaluation for diabetes risk prediction.
Computes demographic parity, equalised odds, and worst‑group accuracy.
Assumes you have a test set with true labels and model predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    MetricFrame
)
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray,
                      sensitive_features: pd.Series,
                      groups: list = None) -> Dict[str, Any]:
    """
    Compute fairness metrics.

    Args:
        y_true: ground truth labels (0/1)
        y_pred: predicted labels (0/1) or probabilities (will be thresholded at 0.5)
        sensitive_features: array of group labels (e.g., age_group, hospital, gender)
        groups: list of group names to report (if None, all groups are considered)

    Returns:
        Dictionary containing:
          - demographic_parity_diff
          - equalized_odds_diff
          - overall_accuracy
          - per_group_accuracy
          - worst_group_accuracy
    """
    # Convert probabilities to binary if needed
    if y_pred.dtype != np.int64 and y_pred.dtype != np.bool_:
        y_pred = (y_pred >= 0.5).astype(int)

    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)

    # Demographic parity difference (difference in selection rates between groups)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)

    # Equalized odds difference (maximum difference in false positive/negative rates)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

    # Per‑group accuracy
    metric_frame = MetricFrame(metrics=accuracy_score,
                               y_true=y_true,
                               y_pred=y_pred,
                               sensitive_features=sensitive_features)
    per_group_acc = metric_frame.by_group.to_dict()
    worst_group_acc = min(per_group_acc.values())

    return {
        "demographic_parity_diff": dp_diff,
        "equalized_odds_diff": eo_diff,
        "overall_accuracy": overall_acc,
        "per_group_accuracy": per_group_acc,
        "worst_group_accuracy": worst_group_acc
    }

# Example usage (to be integrated into your experiment script)
if __name__ == "__main__":
    # This is a placeholder; you would replace with your actual test data.
    # For demo, create synthetic data.
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    # Sensitive feature: age group (<45, 45-65, >65)
    age = np.random.randint(20, 90, n)
    age_group = pd.cut(age, bins=[0,45,65,120], labels=["<45", "45-65", ">65"])
    result = evaluate_fairness(y_true, y_pred, age_group)
    print("Fairness results:")
    for k,v in result.items():
        print(f"{k}: {v}")
