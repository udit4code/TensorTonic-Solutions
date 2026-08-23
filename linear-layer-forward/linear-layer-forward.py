import numpy as np 

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Step 0 : Convert X, W, b to numpy data structures 
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    N, d_in = X.shape 
    _, d_out = W.shape 
    
    # Step 1 : Compute the output 
    output = (X @ W + b)

    return output.tolist()