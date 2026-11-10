import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Splits data into training and testing sets while preserving the percentage 
    of samples for each class (stratified split).
    """
    # Step 1 : Create a random number generator with a fixed seed so results are repeatable
    rng = np.random.default_rng(seed)
    
    # Step 2 : Convert standard Python lists into NumPy arrays for easier manipulation and slicing
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Step 3 : Find all unique class labels (e.g., 0 and 1) in our target labels array
    # np.unique() finds all the unique, distinct values inside an array and returns them sorted from smallest to largest.
    unique_labels = np.unique(y_arr)
    
    train_indices = []
    test_indices = []
    
    # Step 4 : Loop through each class label separately to maintain exact class proportions
    for label in unique_labels:
        # Step 4.1 : Find all positions (indices) where the label matches the current class
        indices = np.flatnonzero(y_arr == label)
        # Step 4.2 : Randomly shuffle the positions for this specific class
        permuted_indices = rng.permutation(indices)
        # Step 4.3 : Calculate how many samples from this class should go into the test set
        n_test = int(np.round(len(permuted_indices) * test_size))
        
        # Step 4.4 : Edge-case safety: ensure at least 1 sample goes to test and train if possible
        if n_test == 0 and len(permuted_indices) > 1 and test_size > 0:
            n_test = 1
        elif n_test == len(permuted_indices) and len(permuted_indices) > 1:
            n_test = len(permuted_indices) - 1
            
        # Step 4.5 : Separate the shuffled indices into test and training groups
        test_indices.append(permuted_indices[:n_test])
        train_indices.append(permuted_indices[n_test:])
        
    # Step 5 : Combine all groups together and sort them in ascending order for a clean sequence
    train_indices = np.sort(np.concatenate(train_indices))
    test_indices = np.sort(np.concatenate(test_indices))
    
    # Step 6 : Use the sorted index arrays to slice our original data and return the dictionary
    return {
        "X_train": X_arr[train_indices],
        "X_test": X_arr[test_indices],
        "y_train": y_arr[train_indices],
        "y_test": y_arr[test_indices]
    }