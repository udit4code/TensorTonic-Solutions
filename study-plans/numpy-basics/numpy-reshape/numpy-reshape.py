import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    np_data = np.array(data, dtype=np.float64)
    if operation == "flatten":
         np_data = np_data.flatten()
    elif operation == "transpose":
        np_data = np_data.T
    elif operation == "add_batch":
        # Alternate way : np_data.reshape((1,) + np_data.shape)
        # It means : Via Tuple addition : (1, ) + (m, n) = (1, m, n)
        np_data = np.expand_dims(np_data, axis=0)
    else:
        raise Exception(f"operation {operation} is invalid.")
    # Why .copy()? So that the returned object does not share the same memory location as np_data
    result = np_data.copy()
    assert id(result) != id(np_data) , f"np_data id : {id(np_data)} vs result id : {id(result)}"
    return result
