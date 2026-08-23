import numpy as np

def tanh(x: list) -> np.ndarray:
    """Return tanh applied elementwise to x."""
    # Step 1 : Convert the input list into a NumPy array
    np_x = np.asarray(x, dtype=np.float64)
    # Step 2 : Compute e^x
    exp_x = np.exp(np_x)

    # Step 3 : Compute e^(-x)
    exp_neg_x = np.exp(-np_x)

    # Step 4 : Apply: Compute tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    output = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

    return output