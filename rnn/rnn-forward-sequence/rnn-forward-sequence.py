import numpy as np

def rnn_forward(X: np.ndarray,h_0: np.ndarray,W_xh: np.ndarray,W_hh: np.ndarray,b_h: np.ndarray):
    """
        Forward pass through a vanilla RNN.
        Args:
            X:      (batch, T, input_dim)
            h_0:    (batch, hidden_dim)
            W_xh:   (input_dim, hidden_dim)
            W_hh:   (hidden_dim, hidden_dim)
            b_h:    (hidden_dim,)
        Returns:
            hidden_states: (batch, T, hidden_dim)
            h_final:       (batch, hidden_dim)
    """
    batch_size, T, _ = X.shape
    hidden_dim = h_0.shape[1]
    hidden_states = np.zeros((batch_size, T, hidden_dim))
    h = h_0.copy() 
    
    for t in range(T):
        x_t = X[:, t, :]                    
        h = np.tanh(x_t @ W_xh.T + h @ W_hh.T + b_h)
        hidden_states[:, t, :] = h

    h_final = h
    return hidden_states, h_final