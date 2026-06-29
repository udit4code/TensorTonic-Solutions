import numpy as np


# This implementation can overflow for very large values of x. 
def tanh(x):
    x = np.asarray(x)
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

# Numerically, more stable : 
# tanh(x) = 2/(1 + exp(-2x)) - 1
def tanh_v2(x): 
    x = np.asarray(x)
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0
    
def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    A = x_t @ W_xh.T + h_prev @ W_hh.T + b_h 
    h_curent = tanh(A) 
    return h_curent