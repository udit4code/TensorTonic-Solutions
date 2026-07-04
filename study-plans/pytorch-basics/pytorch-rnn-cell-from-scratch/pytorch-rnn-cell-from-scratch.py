import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
            A vanilla RNN cell implemented from first principles.
    
            Args:
                input_size: Number of input features.
                hidden_size: Size of the hidden state.
    
            Returns:
                None
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable parameters for the input transformation.
        # Computes: x @ W_ih.T + b_ih
        # W_ih shape: (hidden_size, input_size)
        # b_ih shape: (hidden_size,)
        self.W_ih = nn.Parameter(
            torch.randn(hidden_size, input_size)
        )
        self.b_ih = nn.Parameter(
            torch.zeros(hidden_size)
        )
        # Learnable parameters for the hidden-state transformation.
        # Computes: h_prev @ W_hh.T + b_hh
        # W_hh shape: (hidden_size, hidden_size)
        # b_hh shape: (hidden_size,)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev):
        """
        Args:
            x:
                Current input.
                Shape:
                    (batch_size, input_size)

            h_prev:
                Previous hidden state.
                Shape:
                    (batch_size, hidden_size)

        Returns:
            New hidden state.
            Shape:
                (batch_size, hidden_size)
        """
        # Input transformation:
        # x (whose shape is (batch,input_size)) @ W_ih.T (input_size,hidden_size) -> (batch,hidden_size)
        input_term = x @ self.W_ih.T + self.b_ih
        # Hidden-state transformation:
        # h_prev (batch,hidden_size) @ W_hh.T (hidden_size,hidden_size) ->(batch,hidden_size)
        hidden_term = h_prev @ self.W_hh.T + self.b_hh
        # Combine the current input and previous hidden state, then apply the tanh activation.
        # h_t = tanh(input_term + hidden_term)
        h_new = torch.tanh(input_term + hidden_term)
        return h_new