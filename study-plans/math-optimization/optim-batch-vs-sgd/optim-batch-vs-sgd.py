import numpy as np

def run_batch_gradient_descent(X_arr, y_arr, batch_size, n_epochs, lr, seed, n_samples, n_features):
    # Step 1: Restart weights at zero for each batch size
    w = np.zeros(n_features)        
    
    # Step 2: Restart NumPy RandomState with the supplied seed
    rng = np.random.RandomState(seed)
        
    loss_curve = []

    # Step 3 : Run Epochs
    for epoch in range(n_epochs):
        # Step 3.1 : Shuffle row indices once at the beginning of every epoch
        indices = rng.permutation(n_samples)
            
        # Step 3.2 : Process consecutive batches
        for i in range(0, n_samples, batch_size):
            # Step 3.2.1 : Get Batch indices and then, extract the batch via indexing 
            batch_indices = indices[i : i + batch_size]
            
            X_batch = X_arr[batch_indices]
            y_batch = y_arr[batch_indices]
                
            # Step 3.2.2 : Compute batch predictions
            y_pred_batch = X_batch @ w
                
            # Step 3.2.3 : Compute gradient of the mean squared error loss
            grad = 2 * (X_batch.T @ (y_pred_batch - y_batch)) / len(batch_indices)
                
            # Step 3.2.4 : Update weights
            w -= lr * grad
                
        # Step 3.3 : Record the full-dataset MSE after all updates in the epoch
        y_pred_full = X_arr @ w
        mse = np.mean((y_pred_full - y_arr) ** 2)
        loss_curve.append(float(mse))

    return loss_curve


def batch_gd_compare(X: list, y: list, batch_sizes: list, n_epochs: int, lr: float, seed: int) -> list:
    """
    Returns one post-epoch MSE loss curve per batch size.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    n_samples, n_features = X_arr.shape
    
    all_curves = []

    for batch_size in batch_sizes:
        loss_curve = run_batch_gradient_descent(
            X_arr, y_arr, batch_size, n_epochs, lr, seed, n_samples, n_features
        )
        all_curves.append(loss_curve)

    return all_curves