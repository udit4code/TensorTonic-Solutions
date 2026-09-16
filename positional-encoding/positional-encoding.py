import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    T, d = int(seq_len), int(d_model)
    pos = np.arange(T, dtype=float)[:, None]
    i = np.arange((d + 1) // 2, dtype=float)[None, :]
    div = np.power(base, (2 * i) / d)
    angles = pos / div
    pe = np.zeros((T, d), dtype=float)
    pe[:, 0::2] = np.sin(angles[:, :(d + 1) // 2])[:, :len(pe[0, 0::2])]
    pe[:, 1::2] = np.cos(angles[:, :d // 2])[:, :len(pe[0, 1::2])]
    return pe

    
    