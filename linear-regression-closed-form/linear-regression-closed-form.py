import numpy as np

# The numpy.linalg.solve() function computes the exact solution of a well-determined system of linear equations in the matrix form Ax = b
def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    product_1 = X.T @ X 
    product_2 = X.T @ y 
    return np.linalg.solve(product_1, product_2)
    
def linear_regression_closed_form_v1(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    product_1 = X.T @ X 
    product_2 = X.T @ y 
    # Not numerically stable 
    inv_product_1 = np.linalg.inv(product_1)
    w = inv_product_1 @ product_2
    return w