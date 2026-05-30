import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    result = np.full(shape=(len(seqs), max_len),fill_value=pad_value,dtype=int)

    for i, seq in enumerate(seqs):
        seq = seq[:max_len]  # truncate if longer
        result[i, :len(seq)] = seq

    return result