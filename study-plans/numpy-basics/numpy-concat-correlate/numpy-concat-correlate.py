import numpy as np

def get_pearson_corr_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    # Number of observations.
    m = X.shape[0]
    # Step 1: Compute column means.
    # Example: [[1,2], [2,4], [3,6]] -> [(1 + 2 + 3)/3,(2 + 4 + 6)/3] = [2, 4]
    means = np.mean(X, axis=0)
    # Step 2: Center each column.
    # Example : [[1,2], [2,4], [3,6]] - [2,4] = Via broadcasting, [[2, 4], [2, 4], [2, 4]]
    # = [[1,2], [2,4], [3,6]] - [[2,4], [2,4], [2 4]]
    # = [[-1,-2],[0,0],[1,2]]
    centered = X - means
    # Step 3: Covariance matrix.
    # Cov = X_centered^T X_centered /(m-1), whose shape is (n,n)
    covariance = centered.T @ centered / (m - 1)
    # Step 4: Standard deviations of columns = sqrt(diagonal(covariance))
    std = np.sqrt(np.diag(covariance))
    # Step 5: Outer product of standard deviations.
    # Example: std=[1,2] -> Product of a 2x1 column vector with a 1x2 row vector 
    # [[1], [2]] x [1, 2]] -> [[1,2], [2,4]]
    row = std.reshape(1, -1)
    col = std.reshape(-1, 1)
    denom = col @ row # np.outer(std, std)
    # Step 6:
    # Pearson correlation matrix.
    correlation = covariance / denom
    return correlation

def compare_correlations(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    a_data = np.array(a, dtype=np.float64) # shape (m_1, n)
    b_data = np.array(b, dtype=np.float64) # shape (m_2, n)
    # Now, concatenate row_wise to produce a matrix of shape (m_1 + m_2, n)
    combined_data = np.concatenate([a_data, b_data], axis=0)
    # Each of the below matrices is of the shape (n, n)
    pc_a_data = get_pearson_corr_matrix(a_data)
    pc_b_data = get_pearson_corr_matrix(b_data)
    pc_combined_data = get_pearson_corr_matrix(combined_data)
    return np.stack([pc_a_data, pc_b_data, pc_combined_data])
    
def compare_correlations_v1(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    a_data = np.array(a, dtype=np.float64) # shape (m_1, n)
    b_data = np.array(b, dtype=np.float64) # shape (m_2, n)
    # Now, concatenate row_wise to produce a matrix of shape (m_1 + m_2, n)
    combined_data = np.concatenate([a_data, b_data], axis=0)
    # Each of the below matrices is of the shape (n, n)
    pc_a_data = np.corrcoef(a_data.T)
    pc_b_data =np.corrcoef(b_data.T)
    pc_combined_data = np.corrcoef(combined_data.T)
    return np.stack([pc_a_data, pc_b_data, pc_combined_data])