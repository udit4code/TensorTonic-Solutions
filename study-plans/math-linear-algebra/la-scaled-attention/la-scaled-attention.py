import numpy as np

def scaled_dot_product_attention(Q: list, K: list, V: list) -> np.ndarray:
    """
    Returns the scaled-attention output as a float64 array.
    """
    # Step 1 : Convert Q, K, V into numpy tensors
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    # Step 2 : Compute the dimensionality of the key vectors (d_k)
    N, d_k = K.shape
    # Step 3 : Compute attention scores (Q * K^T / sqrt(d_k))
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)

    # Step 4 : Apply softmax along the last dimension (with max subtraction for numerical stability)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5 : Compute the final output by multiplying attention weights with V
    output = attention_weights @ V

    return output