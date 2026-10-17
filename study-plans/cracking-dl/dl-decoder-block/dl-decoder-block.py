import numpy as np

class MultiHeadAttention:
    def __init__(self, W_q: list | np.ndarray, W_k: list | np.ndarray, W_v: list | np.ndarray, W_o: list | np.ndarray, num_heads: int):
        self.W_q = np.asarray(W_q, dtype=np.float64)
        self.W_k = np.asarray(W_k, dtype=np.float64)
        self.W_v = np.asarray(W_v, dtype=np.float64)
        self.W_o = np.asarray(W_o, dtype=np.float64)
        self.num_heads = num_heads

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """
        Computes multi-head attention. Handles both self-attention (where Q, K, V have 
        the same sequence length) and cross-attention (where Q has decoder length, 
        and K, V have encoder length).
        """
        # Get dimensions
        # Q shape: (T_q, d_model) -> T_q = sequence length of queries, d_model = hidden feature dimension
        # K shape: (T_k, d_model) -> T_k = sequence length of keys
        # V shape: (T_v, d_model) -> T_v = sequence length of values (T_k always equals T_v)
        T_q, d_model = Q.shape
        T_k = K.shape[0]
        T_v = V.shape[0]

        # Step 1 : Ensure model dimension is evenly divisible by the number of heads
        assert d_model % self.num_heads == 0, f"d_model {d_model} is not divisible by num_heads {self.num_heads}"
        
        # d_h is the dimension allocated to each individual attention head (d_model / num_heads)
        d_h = d_model // self.num_heads

        # Step 2 : Project Q, K, V using PyTorch weight convention (.T layout)
        # W_q shape: (d_model, d_model) -> W_q.T is also (d_model, d_model)
        # Q_proj shape: (T_q, d_model) @ (d_model, d_model) = (T_q, d_model)
        Q_proj = Q @ self.W_q.T 
        K_proj = K @ self.W_k.T  # Shape: (T_k, d_model)
        V_proj = V @ self.W_v.T  # Shape: (T_v, d_model)

        # Step 3 : Reshape each projection to separate features into individual heads
        # Q_proj becomes: (T_q, num_heads, d_h)
        # K_proj becomes: (T_k, num_heads, d_h)
        # V_proj becomes: (T_v, num_heads, d_h)
        Q_proj = Q_proj.reshape(T_q, self.num_heads, d_h)
        K_proj = K_proj.reshape(T_k, self.num_heads, d_h)
        V_proj = V_proj.reshape(T_v, self.num_heads, d_h)

        # Step 4 : Transpose so 'num_heads' is the leading dimension (Axis 0)
        # This allows matrix multiplication to happen independently for each head in parallel.
        # Shape becomes: (num_heads, T_q, d_h) for Q, and (num_heads, T_k, d_h) for K & V
        Q_proj = Q_proj.transpose(1, 0, 2)
        K_proj = K_proj.transpose(1, 0, 2)
        V_proj = V_proj.transpose(1, 0, 2)

        # Step 5 : Compute scaled dot-product attention scores for each head
        scale = np.sqrt(d_h)
        # K_proj transposed for matrix multiplication: (num_heads, d_h, T_k)
        # P shape: (num_heads, T_q, d_h) @ (num_heads, d_h, T_k) = (num_heads, T_q, T_k)
        P = (Q_proj @ K_proj.transpose(0, 2, 1)) / scale

        # Step 6 : Apply attention mask if provided (e.g., causal mask for decoder self-attention)
        if mask is not None:
            mask_arr = np.asarray(mask)
            # Broadcast mask across all heads if its shape is (T_q, T_k)
            if mask_arr.ndim == 2:
                mask_arr = mask_arr[np.newaxis, :, :]
            P = np.where(mask_arr, P, P - 1e9)

        # Step 7 : Apply Stable Softmax across the last dimension (T_k)
        # Subtracting row_max prevents numerical overflow (exploding exponentials)
        row_max = np.max(P, axis=-1, keepdims=True)  # Shape: (num_heads, T_q, 1)
        exp_scores = np.exp(P - row_max)             # Shape: (num_heads, T_q, T_k)
        sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True) # Shape: (num_heads, T_q, 1)

        # Step 8 : Calculate attention probability weights
        # attention_weights shape: (num_heads, T_q, T_k)
        attention_weights = exp_scores / sum_exp_scores

        # Step 9 : Multiply attention weights by Value vectors
        # head_outputs shape: (num_heads, T_q, T_k) @ (num_heads, T_v, d_h) = (num_heads, T_q, d_h)
        head_outputs = attention_weights @ V_proj

        # Step 10 : Combine heads back together
        # 10.1 Transpose back to: (T_q, num_heads, d_h)
        concat_heads = head_outputs.transpose(1, 0, 2)
        # 10.2 Flatten/reshape heads back into single hidden dimension: (T_q, d_model)
        concat_heads = concat_heads.reshape(T_q, d_model)

        # Step 11 : Final output projection matrix multiplication
        # output shape: (T_q, d_model) @ (d_model, d_model).T = (T_q, d_model)
        output = concat_heads @ self.W_o.T

        return {
            "output": output,
            "attention_weights": attention_weights
        }


class LayerNormalization:
    def __init__(self, gamma: list | np.ndarray, beta: list | np.ndarray, eps: float = 1e-5):
        self.gamma = np.asarray(gamma, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies Layer Normalization across the feature dimension (axis=-1).
        Formula: ((x - mean) / sqrt(var + eps)) * gamma + beta
        """
        # x shape: (T, d_model)
        # Step 1: Compute mean across features for each token. Shape: (T, 1)
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # Step 2: Compute variance across features for each token. Shape: (T, 1)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Step 3: Normalize to zero mean and unit variance. Shape: (T, d_model)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        # Step 4: Scale and shift using learnable parameters gamma and beta. Shape: (T, d_model)
        output = x_normalized * self.gamma + self.beta
        
        return output


class FeedForwardNetwork:
    def __init__(self, W1: list | np.ndarray, b1: list | np.ndarray, W2: list | np.ndarray, b2: list | np.ndarray):
        self.W1 = np.asarray(W1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.W2 = np.asarray(W2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies Position-wise Feed-Forward Network: Linear -> ReLU -> Linear.
        """
        # x shape: (T, d_model)
        # Step 1: First linear expansion layer (usually expands d_model to 4 * d_model)
        # linear1 shape: (T, d_model) @ (d_hidden, d_model).T + (d_hidden,) = (T, d_hidden)
        linear1 = x @ self.W1.T + self.b1
        
        # Step 2: Apply ReLU non-linearity (element-wise clamping at 0)
        activated = np.maximum(0.0, linear1)
        
        # Step 3: Second projection layer mapping back down to d_model
        # output shape: (T, d_hidden) @ (d_model, d_hidden).T + (d_model,) = (T, d_model)
        output = activated @ self.W2.T + self.b2
        
        return output


class TransformerDecoderBlock:
    def __init__(
        self, 
        W_q1: list, W_k1: list, W_v1: list, W_o1: list, 
        W_q2: list, W_k2: list, W_v2: list, W_o2: list, 
        num_heads: int, 
        W1: list, b1: list, W2: list, b2: list, 
        gamma1: list, beta1: list, 
        gamma2: list, beta2: list, 
        gamma3: list, beta3: list
    ):
        # Sub-layer 1: Masked Self-Attention components
        self.self_attn = MultiHeadAttention(W_q1, W_k1, W_v1, W_o1, num_heads)
        self.norm1 = LayerNormalization(gamma1, beta1)
        
        # Sub-layer 2: Cross-Attention components
        self.cross_attn = MultiHeadAttention(W_q2, W_k2, W_v2, W_o2, num_heads)
        self.norm2 = LayerNormalization(gamma2, beta2)
        
        # Sub-layer 3: Feed-Forward Network components
        self.ffn = FeedForwardNetwork(W1, b1, W2, b2)
        self.norm3 = LayerNormalization(gamma3, beta3)

    def forward(self, x: np.ndarray, enc_output: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """
        Executes the full forward pass of the Transformer Decoder Block.
        """
        # 1. Masked Multi-Head Self-Attention (Q=x, K=x, V=x)
        sa_dict = self.self_attn.forward(x, x, x, mask)
        sa_output = sa_dict["output"]
        
        # 2. First Residual Connection + Layer Normalization
        norm1_output = self.norm1.forward(x + sa_output)
        
        # 3. Multi-Head Cross-Attention (Q from decoder norm1_output, K & V from encoder enc_output)
        ca_dict = self.cross_attn.forward(norm1_output, enc_output, enc_output, mask=None)
        ca_output = ca_dict["output"]
        
        # 4. Second Residual Connection + Layer Normalization
        norm2_output = self.norm2.forward(norm1_output + ca_output)
        
        # 5. Position-wise Feed-Forward Network
        ffn_output = self.ffn.forward(norm2_output)
        
        # 6. Third Residual Connection + Final Layer Normalization
        final_output = self.norm3.forward(norm2_output + ffn_output)
        
        return {
            "self_attn": sa_output,
            "norm1": norm1_output,
            "cross_attn": ca_output,
            "norm2": norm2_output,
            "ffn_output": ffn_output,
            "output": final_output
        }


def decoder_block(
    x: list, 
    enc_output: list, 
    W_q1: list, W_k1: list, W_v1: list, W_o1: list, 
    W_q2: list, W_k2: list, W_v2: list, W_o2: list, 
    num_heads: int, 
    W1: list, b1: list, W2: list, b2: list, 
    gamma1: list, beta1: list, 
    gamma2: list, beta2: list, 
    gamma3: list, beta3: list, 
    mask: list | None = None
) -> dict:
    """
    Functional wrapper adhering to the original signature while leveraging the 
    object-oriented TransformerDecoderBlock architecture under the hood.
    Returns a dictionary of nested lists rounded to 4 decimal places.
    """
    # Initialize the Decoder Block object
    block = TransformerDecoderBlock(
        W_q1, W_k1, W_v1, W_o1, 
        W_q2, W_k2, W_v2, W_o2, 
        num_heads, 
        W1, b1, W2, b2, 
        gamma1, beta1, 
        gamma2, beta2, 
        gamma3, beta3
    )
    
    # Convert inputs to NumPy arrays
    x_arr = np.asarray(x, dtype=np.float64)
    enc_arr = np.asarray(enc_output, dtype=np.float64)
    
    # Run forward pass
    results = block.forward(x_arr, enc_arr, mask)
    
    # Helper to convert numpy arrays back to lists of rounded floats for test compatibility
    format_transform = lambda arr: [[round(float(v), 4) for v in row] for row in arr]
    
    return {
        "self_attn": format_transform(results["self_attn"]),
        "norm1": format_transform(results["norm1"]),
        "cross_attn": format_transform(results["cross_attn"]),
        "norm2": format_transform(results["norm2"]),
        "ffn_output": format_transform(results["ffn_output"]),
        "output": format_transform(results["output"]),
    }