import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Step 1 : Compute the pre-Tanh output
    pre_tanh = x_t @ Wx + h_prev @ Wh + b  
    # Step 2 : Apply Tanh 
    h_t = np.tanh(pre_tanh)

    return h_t
