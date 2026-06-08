import numpy as np

import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute least-squares solution using Singular Value Decomposition (SVD).
    """

    # Convert inputs to float64 for numerical stability.
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Full SVD decomposition X = U Σ Vᵀ
    # U shape      : (m, n)
    # singular_vals: (n,)
    # Vt shape     : (n, n)
    U, singular_vals, Vt = np.linalg.svd(X, full_matrices=False)
    # Compute Σ⁻¹.
    # Since Σ is diagonal, inversion is simply
    # taking reciprocal of each singular value.
    sigma_inv = np.diag(1.0 / singular_vals)
    # Apply: w = V Σ⁻¹ Uᵀ y
    w = Vt.T @ sigma_inv @ U.T @ y
    return w
    
def linear_regression_closed_form_v3(X, y):
    """
    Compute the optimal weight vector using QR decomposition.
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