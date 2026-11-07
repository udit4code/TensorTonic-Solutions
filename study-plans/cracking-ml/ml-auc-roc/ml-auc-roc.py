import numpy as np

def _roc_curve_stats(y_true: list, y_scores: list) -> tuple:
    """
    Helper function to compute FPR and TPR lists across distinct score thresholds.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort scores in descending order and reorder y_true accordingly
    desc_indices = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_indices]
    y_scores = y_scores[desc_indices]
    
    # Find indices where the score changes (distinct thresholds)
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.concatenate([distinct_value_indices, [len(y_scores) - 1]])
    
    # Cumulative true positives and false positives at each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]
    
    total_positives = np.sum(y_true)
    total_negatives = np.sum(1 - y_true)
    
    # Handle edge case where all labels are 0 or all are 1
    if total_positives == 0 or total_negatives == 0:
        return [0.0, 1.0], [0.0, 1.0]
        
    tprs = (tps / total_positives).tolist()
    fprs = (fps / total_negatives).tolist()
    
    # Prepend the starting point (0.0, 0.0) for the ROC curve
    return [0.0] + fprs, [0.0] + tprs


def get_true_positive_rate_list(y_true: list, y_scores: list) -> list:
    """
    Returns the list of True Positive Rates (TPR) corresponding to the ROC curve.
    """
    _, tprs = _roc_curve_stats(y_true, y_scores)
    return tprs


def auc_roc(y_true: list, y_scores: list) -> tuple:
    """
    Returns false-positive rates, true-positive rates, and AUC.
    """
    false_positive_rate_list, true_positive_rate_list = _roc_curve_stats(y_true, y_scores)
    
    # Compute Area Under the Curve using the trapezoidal rule (version-safe)
    x = np.array(false_positive_rate_list)
    y = np.array(true_positive_rate_list)
    area_under_curve = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))
    
    return (
        false_positive_rate_list,
        true_positive_rate_list,
        area_under_curve
    )