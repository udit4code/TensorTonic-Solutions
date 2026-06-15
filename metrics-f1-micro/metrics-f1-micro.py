import numpy as np 

def get_TP(y_true, y_pred):
    # Get Total True positives, which means x is truly positive and also got reported as positive
    unique_classes = set(y_true) | set(y_pred) # Set Union 
    tp_val = 0
    for class_label in unique_classes:
        for y_true_value, y_pred_value in zip(y_true, y_pred):
            if y_pred_value == class_label and y_true_value == class_label:
                tp_val += 1
    return tp_val

def get_FP(y_true, y_pred):
    # Get Total False positives (Means, x is truly negative, but got predicted as positive)
    unique_classes = set(y_true) | set(y_pred) # Set Union 
    fp_val = 0
    for class_label in unique_classes:
        for y_true_value, y_pred_value in zip(y_true, y_pred):
            if y_pred_value == class_label and y_true_value != class_label:
                fp_val += 1
    return fp_val

def get_FN(y_true, y_pred):
    # Get Total False Negatives (Means, x is truly positive, but got predicted falsely as negative)
    unique_classes = set(y_true) | set(y_pred) # Set Union 
    fn_val = 0
    for class_label in unique_classes:
        for y_true_value, y_pred_value in zip(y_true, y_pred):
            if y_pred_value != class_label and y_true_value == class_label:
                fn_val += 1
    return fn_val
    
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    assert len(y_pred) == len(y_true), "Len of y_true and y_pred are different"
    # Step 1 : Get TP
    tp = get_TP(y_true, y_pred)
    # Step 2 : Get FP
    fp = get_FP(y_true, y_pred)
    # Step 3 : Get FN
    fn = get_FN(y_true, y_pred)
    # Compute f1_micro
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    f1_micro_val = 2 * tp / denominator
    return f1_micro_val
    
    