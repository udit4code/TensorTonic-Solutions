import numpy as np


def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using QR decomposition.

    We want to solve the least-squares problem:

        min ||Xw - y||²

    Instead of solving:

        (XᵀX)w = Xᵀy

    we factorize X as:

        X = QR

    where:
        Q -> orthogonal matrix (QᵀQ = I)
        R -> upper-triangular matrix

    Substituting:

        QRw = y

    Multiplying both sides by Qᵀ:

        Rw = Qᵀy

    Since R is upper-triangular, we can efficiently solve for w
    using a linear solver without explicitly computing any matrix inverse.
    """

    # Convert inputs to contiguous float64 NumPy arrays.
    # float64 is preferred for numerical stability in linear algebra.
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # QR decomposition: X = QR
    # Q shape: (n_samples, n_features)
    # R shape: (n_features, n_features)
    # Using QR avoids forming XᵀX, which can amplify numerical errors.
    Q, R = np.linalg.qr(X)
    # Compute the right-hand side of: Rw = Qᵀy
    # Qᵀ projects y into the orthogonal basis defined by Q.
    rhs = Q.T @ y
    # Solve: Rw = rhs
    # R is upper-triangular, so this is more stable than w = inv(R) @ rhs
    # Never explicitly compute matrix inverses when a solver exists.
    w = np.linalg.solve(R, rhs)
    return w
    
# The numpy.linalg.solve() function computes the exact solution of a well-determined system of linear equations in the matrix form Ax = b
def linear_regression_closed_form_v2(X, y):
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