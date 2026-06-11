import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    g = np.array(g, dtype=np.float64)
    s = np.array(s, dtype=np.float64)
    w = np.array(w, dtype=np.float64)
    # Step 1 : Update Running Average
    s = beta * s + (1 - beta) * g * g 
    # Step 2 : Parameter update
    w = w - (lr / (np.sqrt(s + eps))) * g
    return w, s