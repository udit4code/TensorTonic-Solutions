import numpy as np

def kfold_cv(X: list, y: list, model_fn, k: int = 5, seed: int = 42) -> tuple:
    """
    Returns fold accuracies and their mean.
    """
    # Step 1 : Convert X and y to numpy arrays 
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    n_samples, d = X.shape
    indices = np.arange(n_samples)
    
    # Step 2 : Shuffle row indices once using a NumPy RandomState
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    # Step 3 : Create folds with np.array_split
    folds = np.array_split(indices, k)
    
    fold_accuracies = []

    # Step 4 : Computer Fold Accuracies
    for i in range(k):
        # Step 4.1 : Validation indices for the current fold i
        val_indices = folds[i]
        
        # Step 4.2 : Training indices: concatenate all other folds in ascending order
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Step 4.3 : Split data into training and validation sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Step 4.4 : Call model_fn on training arrays to obtain a prediction function
        predict_fn = model_fn(X_train, y_train)
        
        # Step 4.5 : Evaluate the prediction function on fold i (validation set)
        predictions = predict_fn(X_val)
        
        # Step 4.6 : Compute accuracy for the fold
        accuracy = np.mean(predictions == y_val)
        fold_accuracies.append(float(accuracy.item()))
        
    mean_accuracy = float(np.mean(fold_accuracies).item())
    
    return fold_accuracies, mean_accuracy