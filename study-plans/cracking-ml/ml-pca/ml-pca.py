import numpy as np

def pca(X, n_components=2):
    """
    Returns: tuple of (transformed_data, explained_variance_ratios)
    """
    X = np.array(X, dtype=np.float64)
    N, d = X.shape 
    # Step 1 : Center the data 
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # Step 2 : Compute Covariance Matrix
    covariance_matrix = (1/(N - 1)) * (X_centered.T @ X_centered)
    # Step 3 : Get eigen_values and eigen_vectors of covariance_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Step 4 : Get the top k eigen_vectors, as per their eigen_values
    # Get the indices of eigenvalues in sorted order (decreasing)
    indices = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[indices]
    # Sort the eigenvectors based on the order of eigenvalues (in this case, the second dimension)
    eigenvectors = eigenvectors[:, indices]
    components = eigenvectors[:, :n_components]
    # Step 5 : Project the centered data onto the selected eigenvectors
    X_transformed = X_centered @ components
    # Step 6 : Compute the explained variance ratio for each component
    # The eigenvalues quantify how much variance lies along each direction.
    total_var = np.sum(eigenvalues)
    explained = eigenvalues[:n_components] / total_var
    return ([[round(float(v), 4) for v in row] for row in X_transformed],
            [round(float(v), 4) for v in explained])
    
    
