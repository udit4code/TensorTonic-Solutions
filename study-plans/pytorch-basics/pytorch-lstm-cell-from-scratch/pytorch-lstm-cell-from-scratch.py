import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        A vanilla LSTM cell implemented from first principles.

        Returns:
            None
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

       
        # Input Gate
        #
        # Controls how much of the candidate information should be
        # written into the cell state.
        ###########################################################

        self.W_ii = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        self.b_hi = nn.Parameter(torch.zeros(hidden_size))

        ###########################################################
        # Forget Gate
        #
        # Controls how much of the previous cell state should be
        # retained.
        ###########################################################

        self.W_if = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_if = nn.Parameter(torch.zeros(hidden_size))
        self.b_hf = nn.Parameter(torch.zeros(hidden_size))

        ###########################################################
        # Cell Candidate
        #
        # Produces new candidate information that may be written
        # into the cell state.
        ###########################################################

        self.W_ig = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_ig = nn.Parameter(torch.zeros(hidden_size))
        self.b_hg = nn.Parameter(torch.zeros(hidden_size))

        ###########################################################
        # Output Gate
        #
        # Controls how much of the cell state becomes the hidden
        # state exposed to the next layer/time step.
        ###########################################################

        self.W_io = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_io = nn.Parameter(torch.zeros(hidden_size))
        self.b_ho = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x:
                (batch_size, input_size)

            h_prev:
                (batch_size, hidden_size)

            c_prev:
                (batch_size, hidden_size)

        Returns:
            (h_t, c_t)
        """

        ###########################################################
        # Input Gate
        #
        # Determines how much new information should enter the cell.
        ###########################################################

        i_t = torch.sigmoid(
            x @ self.W_ii.T +
            self.b_ii +
            h_prev @ self.W_hi.T +
            self.b_hi
        )

        ###########################################################
        # Forget Gate
        #
        # Determines how much previous memory should be retained.
        ###########################################################

        f_t = torch.sigmoid(
            x @ self.W_if.T +
            self.b_if +
            h_prev @ self.W_hf.T +
            self.b_hf
        )

        ###########################################################
        # Cell Candidate
        #
        # Computes new candidate memory values.
        ###########################################################

        g_t = torch.tanh(
            x @ self.W_ig.T +
            self.b_ig +
            h_prev @ self.W_hg.T +
            self.b_hg
        )

        ###########################################################
        # Output Gate
        #
        # Determines how much of the cell state should be exposed
        # as the hidden state.
        ###########################################################

        o_t = torch.sigmoid(
            x @ self.W_io.T +
            self.b_io +
            h_prev @ self.W_ho.T +
            self.b_ho
        )

        ###########################################################
        # Update the cell state.
        #
        # Keep part of the old memory and write new information.
        ###########################################################

        c_t = f_t * c_prev + i_t * g_t

        ###########################################################
        # Compute the hidden state.
        #
        # The output gate filters the activated cell state.
        ###########################################################

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t