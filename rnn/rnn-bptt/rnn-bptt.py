import numpy as np

import numpy as np

def bptt_single_step(dh_next: np.ndarray,
                     h_t: np.ndarray,
                     h_prev: np.ndarray,
                     x_t: np.ndarray,
                     W_hh: np.ndarray):
    """
    Backpropagation through one RNN timestep.

    Args:
        dh_next : (batch, hidden_dim)
        h_t     : (batch, hidden_dim)
        h_prev  : (batch, hidden_dim)
        x_t     : unused here
        W_hh    : (hidden_dim, hidden_dim)

    Returns:
        dh_prev : (batch, hidden_dim)
        dW_hh   : (hidden_dim, hidden_dim)
    """

    da = dh_next * (1 - h_t ** 2)
    dh_prev = da @ W_hh
    dW_hh = da.T @ h_prev
    return dh_prev, dW_hh