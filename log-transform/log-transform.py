import numpy as np

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    return list(map(lambda x: np.log(1 + x), values))
   