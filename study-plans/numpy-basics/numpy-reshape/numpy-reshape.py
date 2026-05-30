import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    np_data = np.array(data, dtype=np.float64)
    if operation == "flatten":
         return np_data.flatten().copy()
    elif operation == "transpose":
        # Why .copy()? So that the returned object does not share the same memory location as np_data
        return np_data.T.copy()
    elif operation == "add_batch":
        # Alternate way : np_data.reshape((1,) + np_data.shape)
        # It means : Via Tuple addition : (1, ) + (m, n) = (1, m, n)
        return np.expand_dims(np_data, axis=0).copy()
    raise Exception(f"operation {operation} is invalid.")
