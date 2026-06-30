import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        batch_size, T, _ = X.shape
        if h_0 is None:
            h = np.zeros((batch_size, self.hidden_dim))
        else:
            h = h_0.copy()

        output_dim = self.W_hy.shape[0]

        y_seq = np.zeros((batch_size, T, output_dim))
        for t in range(T):
            x_t = X[:, t, :]
            h = np.tanh(x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            y = (h @ self.W_hy.T + self.b_y)
            y_seq[:, t, :] = y

        return y_seq, h

    def backward(self, dY: np.ndarray):
        """
        Backpropagation Through Time (BPTT).
    
        Args:
            dY: Gradient of loss w.r.t. outputs.
                Shape: (batch, T, output_dim)
    
        Returns:
            Dictionary containing parameter gradients.
        """
    
        batch_size, T, _ = dY.shape
    
        # Step 0 : Initialize parameter gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
    
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
    
        # Step 1 : Gradient flowing backward through time
        dh_next = np.zeros((batch_size, self.hidden_dim))
    
        # Step 2 : Loop backwards through time
        for t in reversed(range(T)):
    
            h_t = self.hidden_states[:, t, :]
            x_t = self.X[:, t, :]
    
            if t == 0:
                h_prev = self.h0
            else:
                h_prev = self.hidden_states[:, t - 1, :]
    
            # Output layer
            dy = dY[:, t, :]
    
            dW_hy += dy.T @ h_t
            db_y += np.sum(dy, axis=0)
    
            # Gradient arriving at hidden state
            dh = dy @ self.W_hy + dh_next
    
            # tanh derivative
            da = dh * (1 - h_t ** 2)
    
            # Hidden layer gradients
            dW_xh += da.T @ x_t
            dW_hh += da.T @ h_prev
    
            db_h += np.sum(da, axis=0)
    
            # Propagate to previous timestep
            dh_next = da @ self.W_hh
    
        return {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "W_hy": dW_hy,
            "b_h": db_h,
            "b_y": db_y,
        }