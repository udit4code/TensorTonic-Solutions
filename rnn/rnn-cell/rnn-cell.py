import numpy as np


def tanh(x):
    x = np.asarray(x)
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)
    
def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    A = x_t @ W_xh.T + h_prev @ W_hh.T + b_h 
    h_curent = tanh(A) 
    return h_curent