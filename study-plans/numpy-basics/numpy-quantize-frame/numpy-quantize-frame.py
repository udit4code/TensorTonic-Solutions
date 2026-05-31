import numpy as np

def quantize_and_frame_v1(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    np_data = np.asarray(data, dtype=np.float64)
    # Round to the specified number of decimal places.
    rounded = np.round(np_data, decimals=decimals)
    # Floor: largest integer <= element.
    floored = np.floor(np_data)
    # Ceil: smallest integer >= element.
    ceiled = np.ceil(np_data)
    # Add a border of zeros around each matrix.
    rounded = np.pad(rounded, pad_width, mode="constant")
    floored = np.pad(floored, pad_width, mode="constant")
    ceiled = np.pad(ceiled, pad_width, mode="constant")

    return np.stack([rounded, floored, ceiled])


def get_padded_matrix(A, pad_width):
    # Say, A = [[1, 2], [3, 4]]
    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    p = pad_width
    # Create a larger matrix filled with zeros.
    # Example: A.shape = (2,2)
    # p = 1
    # New shape: (2+2, 2+2) = (4,4)
    result = np.zeros((m + 2*p, n + 2*p), dtype=A.dtype)
    # Place the original matrix in the center.
    # Example: result[1:3, 1:3] = A
    # gives:
    # 0 0 0 0
    # 0 1 2 0
    # 0 3 4 0
    # 0 0 0 0
    result[p:p+m, p:p+n] = A
    return result

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    np_data = np.asarray(data, dtype=np.float64)
    # Round each value to the requested number of decimal places.
    # Example:[[1.6, 2.4]] -> [[2.0, 2.0]]
    rounded = np.round(np_data, decimals=decimals)
    # Floor each value.
    # floor(x) = largest integer <= x
    # [[1.6, 2.4]] -> [[1.0, 2.0]]
    floored = np.floor(np_data)
    # Ceil each value.
    # ceil(x) = smallest integer >= x
    # [[1.6, 2.4]] -> [[2.0, 3.0]]
    ceiled = np.ceil(np_data)

    rounded = get_padded_matrix(rounded, pad_width)
    floored = get_padded_matrix(floored, pad_width)
    ceiled = get_padded_matrix(ceiled, pad_width)
    return np.stack([rounded, floored, ceiled])