import numpy as np 

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    result = []
    get_theta = lambda k : 2 * math.pi * k / period
    for v in values:
        angle = get_theta(v)
        result.append([math.sin(angle), math.cos(angle)])
    return result