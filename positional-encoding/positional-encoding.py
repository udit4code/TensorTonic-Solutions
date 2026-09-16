import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    T, d = int(seq_len), int(d_model)
    
    # 1. Create position vector and reshape to column vector: shape (T, 1)
    pos = np.arange(T, dtype=float).reshape(T, 1)
    
    # 2. Compute indices for the even/sine tracks.
    # For odd d, we need (d + 1) // 2 channels to cover the extra sine column.
    num_sin_channels = (d + 1) // 2
    i = np.arange(num_sin_channels, dtype=float).reshape(1, num_sin_channels)
    
    # 3. Compute denominator frequencies: shape (1, num_sin_channels)
    div = np.power(base, (2 * i) / d)
    
    # 4. Broadcast positions against frequencies to compute shared angles: shape (T, num_sin_channels)
    angles = pos / div
    
    # 5. Initialize the output positional encoding matrix: shape (T, d)
    pe = np.zeros((T, d), dtype=float)
    
    # 6. Assign sine to all even column slots (0, 2, 4...)
    # angles already matches the exact size needed for the even slots: (T, num_sin_channels)
    pe[:, 0::2] = np.sin(angles)
    
    # 7. Assign cosine to all odd column slots (1, 3, 5...)
    # The number of odd slots is exactly d // 2. We extract only the required columns 
    # from the shared angles matrix to perfectly match the target slice shape.
    num_cos_channels = d // 2
    pe[:, 1::2] = np.cos(angles[:, :num_cos_channels])
    
    return pe

    
    