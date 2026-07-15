import numpy as np

def convexity_certificate(H):
    """
    Returns: dict with 'is_convex' (bool) and 'min_eigenvalue' (float, rounded to 6 decimals)
    """
    H = np.asarray(H, dtype=float)

    # Ensure H is a square matrix
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Hessian must be a square matrix.")

    # Compute eigenvalues for a symmetric matrix
    eigenvalues = np.linalg.eigvalsh(H)
    min_eig = eigenvalues.min()

    # Small tolerance for numerical precision
    tol = 1e-6

    return {
        "is_convex": bool(min_eig >= -tol),
        "min_eigenvalue": round(float(min_eig), 6),
    }
