import numpy as np

def compare_correlations(a, b):
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