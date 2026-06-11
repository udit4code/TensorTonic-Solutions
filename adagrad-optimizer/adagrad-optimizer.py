import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w = np.array(w, dtype=np.float64)
    g = np.array(g, dtype=np.float64)
    G = np.array(G, dtype=np.float64)
    # Step 1 : Accumulate Squared Gradients
    G = G + g * g 
    # Step 2 : Parameter update
    w = w - (lr/np.sqrt(G + eps)) * g
    return w, G