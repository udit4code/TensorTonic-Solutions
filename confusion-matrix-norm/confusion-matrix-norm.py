import numpy as np

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    # Step 1 : Create a zero matrix, with num_classes x num_classes shape 
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    # Step 2 : Now, for each cell in that matrix indexed by (i, j), do a +1 if y_true = i and y_pred = j across each prediction
    # Naive implementation, with a for loop : 
    # for ground_truth, actual_prediction in zip(y_true, y_pred):
    #     confusion_matrix[ground_truth, actual_prediction] += 1
    # But, we want to do it in vectorized fashion via Numpy 
    if y_true.size > 0 and y_pred.size > 0:
        np.add.at(confusion_matrix, (y_true, y_pred), 1)
    return confusion_matrix 

def confusion_matrix_norm(y_true: list, y_pred: list, num_classes: int | None = None, normalize: str = "none") -> np.ndarray:
    """
    Returns the confusion matrix as a NumPy array.
    """
    # Step 0 : Infer the class count when num_classes in None
    if num_classes is None: 
        num_classes = max(max(y_true), max(y_pred)) + 1
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Step 1 : Get Confusion matrix 
    confusion_matrix = get_confusion_matrix(y_true, y_pred, num_classes)
    # Step 2 : Apply Normalization based on mode 
    if normalize == "none":
        return confusion_matrix 
    elif normalize == "true":
        # In this case, the rows should sum up to 1.
        # Step 2.1 : Compute row sums, keeping dimensions intact: shape (num_classes, 1)
        # Why axis = 1 ? Because, for a chosen row, we want to sum across all columns, which are along axis 1.
        row_sums = np.sum(confusion_matrix, axis=1, keepdims=True)
        # Step 2.2 : Avoid division by zero by replacing 0 sums with 1
        row_sums[row_sums == 0] = 1.0
        # Step 2.3 : Broadcast division across rows
        normalized_confusion_matrix = confusion_matrix / row_sums
        return normalized_confusion_matrix
    elif normalize == "pred":
        # In this case, the columns should sum up to 1.
        # Step 2.1 : Compute colsums, keeping dimensions intact. 
        # Why axis = 0 ? Because, for a chosen column along axis 1, we want to sum across all rows, which are along axis 0.
        col_sums = np.sum(confusion_matrix, axis=0, keepdims=True)
        # Step 2.2 : For edge case, we want to avoid division by zero by replacing 0 sums with 1
        col_sums[col_sums == 0] = 1.0
        # Step 2.3 : Broadcast division across all columns
        normalized_confusion_matrix = confusion_matrix / col_sums
        return normalized_confusion_matrix
    elif normalize == "all":
        # In this case, cells across the entire matrix should sum up to 1. 
        total_sum = np.sum(confusion_matrix)
        # 2. Avoid division by zero if the matrix is entirely zeros
        if total_sum == 0:
            total_sum = 1.0
        # Step 3 :  Divide the matrix by the scalar total sum
        normalized_confusion_matrix = confusion_matrix / total_sum
        return normalized_confusion_matrix
    raise Exception(f"Not yet implemented normalization mode {normalize}.")