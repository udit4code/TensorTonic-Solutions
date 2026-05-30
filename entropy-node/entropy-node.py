import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y, dtype=float)
    
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    probs = counts / counts.sum()

    # Ignore zero probabilities
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)