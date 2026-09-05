import numpy as np

def matrix_factorization_sgd_step(U: list, V: list, r: float, lr: float, reg: float) -> list:
    """
    Returns the updated user and item vectors in a two-item list.
    """
    # Step 1 : Convert U and V into numpy arrays first
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    # Step 2 : Compute dot product of U and V.
    dot_product = U @ V.T 
    # Step 3 : Compute e 
    e = r - dot_product 
    # Step 4 : Update U and V 
    U_new = U + lr * (e * V - reg * U)
    V_new = V + lr * (e * U - reg * V)

    return [
        U_new.round(4).tolist(),
        V_new.round(4).tolist(),
    ]