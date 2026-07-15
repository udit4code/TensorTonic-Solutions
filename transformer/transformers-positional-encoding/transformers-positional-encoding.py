import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Step 1 : Positions: (seq_length, 1)
    positions = np.arange(seq_length)[:, np.newaxis]

    # Step 2 : Even dimensions: (1, d_model//2)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    # Step 3 : Initialize encoding matrix
    pe = np.zeros((seq_length, d_model))

    # Step 4 : Apply sine to even indices
    pe[:, 0::2] = np.sin(positions * div_term)

    # Ste[ 5 : Apply cosine to odd indices
    pe[:, 1::2] = np.cos(positions * div_term)

    return pe