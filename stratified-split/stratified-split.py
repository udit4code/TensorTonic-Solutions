import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test as NumPy arrays.
    """
    rng = np.random.default_rng(seed)
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    unique_labels = np.unique(y_arr)
    
    train_indices = []
    test_indices = []
    
    for label in unique_labels:
        # Get class indices using np.flatnonzero and permute them independently
        indices = np.flatnonzero(y_arr == label)
        permuted_indices = rng.permutation(indices)
        
        # Calculate test count and handle edge cases
        n_test = int(np.round(len(permuted_indices) * test_size))
        if n_test == 0 and len(permuted_indices) > 1 and test_size > 0:
            n_test = 1
        elif n_test == len(permuted_indices) and len(permuted_indices) > 1:
            n_test = len(permuted_indices) - 1
            
        test_indices.append(permuted_indices[:n_test])
        train_indices.append(permuted_indices[n_test:])
        
    # Accumulate and sort combined indices before indexing
    train_indices = np.sort(np.concatenate(train_indices))
    test_indices = np.sort(np.concatenate(test_indices))
    
    return {
        "X_train": X_arr[train_indices],
        "X_test": X_arr[test_indices],
        "y_train": y_arr[train_indices],
        "y_test": y_arr[test_indices]
    }