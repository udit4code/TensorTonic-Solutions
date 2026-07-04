import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi-Head Scaled Dot-Product Attention.

        Args:
            d_model: Embedding dimension.
            num_heads: Number of attention heads.

        Returns:
            None
        """
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        ###########################################################
        # Learnable projection matrices.
        #
        # Each has shape:
        #
        # (d_model, d_model)
        #
        # They project the input embeddings into
        # Query, Key, Value and Output spaces.
        ###########################################################

        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V:
                Shape:
                    (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape:
                (batch_size, seq_len, d_model)
        """

        batch_size, seq_len, _ = Q.shape

        ###########################################################
        # Step 1
        #
        # Project the inputs into
        #
        # Query
        # Key
        # Value
        #
        # spaces.
        #
        # Shape remains:
        #
        # (batch, seq_len, d_model)
        ###########################################################

        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v

        ###########################################################
        # Step 2
        #
        # Split the embedding dimension into multiple heads.
        #
        # Before:
        #
        # (B, S, d_model)
        #
        # After reshape:
        #
        # (B, S, heads, head_dim)
        #
        # Then transpose:
        #
        # (B, heads, S, head_dim)
        ###########################################################

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        ###########################################################
        # Step 3
        #
        # Compute attention scores.
        #
        # Shape:
        #
        # (B, heads, S, S)
        #
        # Each query is compared against every key.
        ###########################################################

        scores = Q @ K.transpose(-2, -1)

        ###########################################################
        # Step 4
        #
        # Scale the attention scores.
        #
        # This prevents very large dot products from causing
        # extremely peaked softmax distributions.
        ###########################################################

        scores = scores / math.sqrt(self.head_dim)

        ###########################################################
        # Step 5
        #
        # Convert attention scores into probabilities.
        #
        # Softmax is applied across the key dimension.
        ###########################################################

        attention = torch.softmax(scores, dim=-1)

        ###########################################################
        # Step 6
        #
        # Compute weighted sums of the value vectors.
        #
        # Shape:
        #
        # (B, heads, S, head_dim)
        ###########################################################

        output = attention @ V

        ###########################################################
        # Step 7
        #
        # Concatenate all heads.
        #
        # Before:
        #
        # (B, heads, S, head_dim)
        #
        # After:
        #
        # (B, S, d_model)
        ###########################################################

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        # Step 8
        # Apply the final output projection.
        # Shape remains:
        # (B, S, d_model)
        ###########################################################

        output = output @ self.W_o

        return output