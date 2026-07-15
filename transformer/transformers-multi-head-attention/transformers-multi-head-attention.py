import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    head_dim = d_model // num_heads

    assert d_model % num_heads == 0 , f"d_model {d_model} is not divisible by num_heads {num_heads}"
    
    # Linear projections
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    # Split into heads
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    heads = np.matmul(weights, V)

    # Concatenate heads
    heads = heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Output projection
    output = heads @ W_o

    return output