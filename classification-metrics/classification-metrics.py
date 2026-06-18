import numpy as np

def get_confusion_matrix(y_true, y_pred):
    # Say, y_true = [0 1 2 2] and y_pred = [0 1 0 2]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # np.concatenate([y_true, y_pred]) gives [0 1 2 2 0 1 0 2] and then, we apply np.unique([0 1 2 2 0 1 0 2]), which gives us [0, 1, 2] . So, labels = [0, 1, 2]
    labels = np.unique(np.concatenate([y_true, y_pred]))

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    # Now, label_to_idx = {0 : 0, 1 : 1, 2 : 2}.
    # Why ? Because confusion_matrix uses contiguous row/column indices.
    # So, Along the row, label_0 -> row_0, ...., label_2 -> row_2
    # Similarly, along the column, label_0, col_0, ..., label_2 -> col_2
    n_classes = len(labels)
    # Initialise the confusion matrix with a zero matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    # rows = truth and cols = prediction
    # Then, then zip(y_true, y_pred) = [(0, 0), (1, 1), (2, 0), (2, 2)]
    for yt, yp in zip(y_true, y_pred):
        # Say, we have (0, 0) . So, row_idx = [0], col_idx = [0]. 
        # Hence, confusion_matrix[[0], [0]] += 1
        row_idx = label_to_idx[yt]
        col_idx = label_to_idx[yp]
        confusion_matrix[row_idx, col_idx] += 1
    # After the loop, our confusion matrix will be [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    # We can think of a confusion matrix as nothing more than a 2D histogram of (true_class, predicted_class) pairs.
    return confusion_matrix, labels

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    confusion_matrix, labels = get_confusion_matrix(y_true, y_pred)
    # For binary classification : When both ground_truth and predicted_value are same
    # For multiclass classification : tp[k] = number of samples whose true class is k and were predicted as class k
    tp = np.diag(confusion_matrix) 
    # For binary classification : when predicted_value is positive and ground_truth is negative
    # For multiclass classification : fp[k] = number of samples predicted as class k and whose true class was NOT k
    fp = confusion_matrix.sum(axis=0) - tp
    # For binary classification : when predicted_value is negative and ground_truth is positive
    # For multiclass classification : fn[k] = number of samples whose true class is k but were predicted as some other class
    fn = confusion_matrix.sum(axis=1) - tp 
    # trace(cm) = total correctly classified samples , which lie along the main diagonal
    # sum(confusion_matrix)   = total samples or total observations
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) 
    # support[k] = number of examples belonging to class k
    # To get support[k], we count all elements along the row k across all columns.
    support = confusion_matrix.sum(axis=1)

    eps = 1e-12

    if average == "binary":
        if pos_label not in labels:
            raise ValueError(f"pos_label={pos_label} not found in labels")

        pos_idx = np.where(labels == pos_label)[0][0]
        precision = tp[pos_idx] / (tp[pos_idx] + fp[pos_idx] + eps)
        recall = tp[pos_idx] / (tp[pos_idx] + fn[pos_idx] + eps)

        f1 = (2 * precision * recall / (precision + recall + eps))
    elif average == "micro":
        tp_micro = tp.sum()
        fp_micro = fp.sum()
        fn_micro = fn.sum()

        precision = tp_micro / (tp_micro + fp_micro + eps)
        recall = tp_micro / (tp_micro + fn_micro + eps)

        f1 = (2 * precision * recall / (precision + recall + eps))
    elif average == "macro":
        precision_per_class = tp / (tp + fp + eps)
        recall_per_class = tp / (tp + fn + eps)
        f1_per_class = (2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps))

        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1 = f1_per_class.mean()
    elif average == "weighted":

        precision_per_class = tp / (tp + fp + eps)
        recall_per_class = tp / (tp + fn + eps)
        f1_per_class = (2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps))
        precision = np.average(precision_per_class,weights=support)
        recall = np.average(recall_per_class,weights=support)
        f1 = np.average(f1_per_class,weights=support)
    else:
        raise ValueError(
            "average must be one of "
            "['binary', 'micro', 'macro', 'weighted']"
        )
    result = {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
    }
    return result
    