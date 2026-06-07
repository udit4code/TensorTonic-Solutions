import numpy as np

def get_label_encoding(data):
    sorted_class_labels = sorted(list(set(data)))
    class_to_idx_map = { }
    for index, class_label in enumerate(sorted_class_labels):
        class_to_idx_map[class_label] = index 
    return {
        "encoded" : [class_to_idx_map[class_label] for class_label in data],
        "classes" : sorted_class_labels,
    }

def get_onehot_encoding(data):
    class_count = len(set(data))
    result = [ ]
    sorted_class_labels = sorted(list(set(data)))
    class_to_idx_map = { }
    for index, class_label in enumerate(sorted_class_labels):
        class_to_idx_map[class_label] = index 
    for class_label in data:
        vector = [0] * class_count 
        vector[class_to_idx_map[class_label]] = 1
        result.append(vector)
    return result 
    
def categorical_encode(data, method="label"):
    """
    Returns: encoded result based on method
    """
    result = None
    if method == "label":
        result = get_label_encoding(data)
    elif method == "onehot":
        result = get_onehot_encoding(data) 
    else:
        raise Exception(f"invalid method : {method}")
    return result