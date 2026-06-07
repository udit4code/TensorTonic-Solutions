import numpy as np

import numpy as np

def batch_norm(X, gamma, beta, running_mean, running_var, mode):
    """
    Returns: dict with keys "output", "running_mean", "running_var"
    """
    X = np.array(X, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    rm = np.array(running_mean, dtype=float)
    rv = np.array(running_var, dtype=float)
    eps = 1e-5
    momentum = 0.1

    if mode == "train":
        mu = X.mean(axis=0)
        var = ((X - mu) ** 2).mean(axis=0)
        X_hat = (X - mu) / np.sqrt(var + eps)
        out = gamma * X_hat + beta
        rm = (1 - momentum) * rm + momentum * mu
        rv = (1 - momentum) * rv + momentum * var
    else:
        X_hat = (X - rm) / np.sqrt(rv + eps)
        out = gamma * X_hat + beta

    return {
        "output": [[round(float(v), 4) for v in row] for row in out],
        "running_mean": [round(float(v), 4) for v in rm],
        "running_var": [round(float(v), 4) for v in rv]
    }

