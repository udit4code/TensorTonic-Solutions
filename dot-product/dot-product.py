import numpy as np

def dot_product(x: list, y: list) -> float:
    """Return the dot product of x and y."""
    # Step 1 : Convert x and y to numpy data structures with initial shape (d,)
    np_x = np.asarray(x, dtype=np.float64)
    np_y = np.asarray(y, dtype=np.float64)
    # Step 2 : Reshape np_x and np_y to (1, d) shape 
    np_x = np.reshape(np_x, (1, -1))
    np_y = np.reshape(np_y, (1, -1))
    # Step 3 : Do a matrix multiplication np_x @ np_y.T [(1,d) @ (d, 1)] and then, reduce it over the axis 0
    output = np.sum(np_x @ np_y.T, axis=0)

    return output.item()