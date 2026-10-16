import numpy as np

class MultiHeadAttention:
    def __init__(self, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int):
        self.W_q = np.asarray(W_q, dtype=np.float64)
        self.W_k = np.asarray(W_k, dtype=np.float64)
        self.W_v = np.asarray(W_v, dtype=np.float64)
        self.W_o = np.asarray(W_o, dtype=np.float64)
        self.num_heads = num_heads

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: list | None = None) -> dict:
        """
        Computes multi-head attention.
        """
        T, d_model = Q.shape
        
        # Step 1 : Get dimension of each head (Each head consists of unique key, value and query head in this case)
        assert d_model % self.num_heads == 0, f"d_model {d_model} is not divisible by num_heads {self.num_heads}"
        # d_h is dimension of each head.
        d_h = d_model // self.num_heads

        # Step 2 : Compute Linear Projections using .T (Standard Linear Layer convention)
        Q_proj = Q @ self.W_q.T
        K_proj = K @ self.W_k.T 
        V_proj = V @ self.W_v.T 

        # Step 3 & 4 : Reshape and transpose to (num_heads, T, d_h)
        Q_proj = Q_proj.reshape(T, self.num_heads, d_h).transpose(1, 0, 2)
        K_proj = K_proj.reshape(T, self.num_heads, d_h).transpose(1, 0, 2)
        V_proj = V_proj.reshape(T, self.num_heads, d_h).transpose(1, 0, 2)
        # So far, shape of each Q_proj, K_proj, V_proj is (num_heads, T, d_h). 
        # We do this, because we want to compute attention for each head. 

        # Step 5 : Compute Scaled dot product for each head, but in 1 shot
        scale = np.sqrt(d_h)
        K_proj_transposed = K_proj.transpose(0, 2, 1)
        P = Q_proj @ K_proj_transposed / scale
        # Shape of P is (num_heads, T, T)

        # Step 5.1 : [Conditional] Apply mask on P, if provided
        if mask is not None:
            mask_np_arr = np.asarray(mask, dtype=np.float64)
            P = P + mask_np_arr

        # Step 6 & 7 : Stable Softmax and Attention Weights
        row_max = np.max(P, axis=-1, keepdims=True)
        exp_scores = np.exp(P - row_max)
        sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)
        attention_weights = exp_scores / sum_exp_scores

        # Step 8 : Compute head outputs from attention_weights
        # shape of head_outputs is (num_heads, T, d_h)
        head_outputs = attention_weights @ V_proj

        # Step 9 : Concatenate heads back to (T, d_model)
        concat_heads = head_outputs.transpose(1, 0, 2).reshape(T, d_model)
        
        # Step 10 : Multiply by output projection matrix W_o -> Shape: (T, d_model)
        output = concat_heads @ self.W_o.T 

        return {
            "output": output,
            "attention_weights": attention_weights
        }


class LayerNormalization:
    def __init__(self, gamma: list, beta: list, eps: float = 1e-5):
        self.gamma = np.asarray(gamma, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies Layer Normalization to the input tensor.
        Formula: LN(x) = (x - mean) / sqrt(variance + eps) * gamma + beta
        """
        # Step 1: Compute the mean across the feature dimension (axis=-1).
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # Step 2: Compute the variance across the feature dimension (axis=-1).
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Step 3: Normalize the input (zero mean, unit variance)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        # Step 4: Scale by learnable gamma and shift by learnable beta.
        output = x_normalized * self.gamma + self.beta
        
        return output


class FeedForwardNetwork:
    def __init__(self, W1: list, b1: list, W2: list, b2: list):
        self.W1 = np.asarray(W1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.W2 = np.asarray(W2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies Position-wise Feed-Forward Network (Linear -> ReLU -> Linear).
        Formula: FFN(x) = max(0, x @ W1.T + b1) @ W2.T + b2
        """
        # Step 1: First linear transformation (expansion layer)
        linear1 = x @ self.W1.T + self.b1
        
        # Step 2: Apply ReLU non-linearity
        activated = np.maximum(0.0, linear1)
        
        # Step 3: Second linear transformation (projection layer back to d_model)
        output = activated @ self.W2.T + self.b2
        
        return output


class TransformerEncoderBlock:
    def __init__(self, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, 
                 W1: list, b1: list, W2: list, b2: list, 
                 gamma1: list, beta1: list, gamma2: list, beta2: list):
        self.mha = MultiHeadAttention(W_q, W_k, W_v, W_o, num_heads)
        self.norm1 = LayerNormalization(gamma1, beta1)
        self.ffn = FeedForwardNetwork(W1, b1, W2, b2)
        self.norm2 = LayerNormalization(gamma2, beta2)

    def forward(self, x: list, mask: list | None = None) -> dict:
        x_arr = np.asarray(x, dtype=np.float64)
        
        # 1. Multi-Head Self-Attention (Q=x, K=x, V=x)
        attn_dict = self.mha.forward(x_arr, x_arr, x_arr, mask)
        attn_output = attn_dict["output"]
        
        # 2. First Residual Connection + Layer Normalization (Post-LN Transformer Block)
        norm1_input = x_arr + attn_output
        norm1_output = self.norm1.forward(norm1_input)
        
        # 3. Position-wise Feed-Forward Network
        ffn_output = self.ffn.forward(norm1_output)
        
        # 4. Second Residual Connection + Layer Normalization
        norm2_input = norm1_output + ffn_output
        final_output = self.norm2.forward(norm2_input)

        transform = lambda arr: [[round(float(v), 4) for v in row] for row in arr]
        return {
            "attention_output": transform(attn_output.tolist()),
            "norm1": transform(norm1_output.tolist()),
            "ffn_output": transform(ffn_output.tolist()),
            "output": transform(final_output.tolist()),
        }


def encoder_block(x: list, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, 
                  W1: list, b1: list, W2: list, b2: list, 
                  gamma1: list, beta1: list, gamma2: list, beta2: list) -> dict:
    """
    Returns a dictionary of lists: attention_output, norm1, ffn_output, output.
    """
    block = TransformerEncoderBlock(
        W_q, W_k, W_v, W_o, num_heads, 
        W1, b1, W2, b2, 
        gamma1, beta1, gamma2, beta2
    )
    return block.forward(x)
