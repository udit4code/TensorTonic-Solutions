import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    product_1 = X.T @ X 
    product_2 = X.T @ y 
    inv_product_1 = np.linalg.inv(product_1)
    w = inv_product_1 @ product_2
    return w