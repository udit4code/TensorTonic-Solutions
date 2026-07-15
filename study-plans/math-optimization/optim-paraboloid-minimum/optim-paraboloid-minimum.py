def paraboloid_minimum(a, b, c, d, e):
    """
    Returns: dict with 'x_star', 'y_star', 'f_min' (floats), each rounded to 6 decimals
    """
    x_star = -(c /(2 * a))
    y_star = -(d /(2 * b))
    f_min = a * x_star * x_star + b * y_star * y_star + c * x_star + d * y_star + e 
    return {
        "x_star" : x_star, 
        "y_star" : y_star,
        "f_min" : f_min,
    }
