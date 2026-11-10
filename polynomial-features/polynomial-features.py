def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """
    result = [ ]
    for value in values:
        acc = 1
        d = 0
        feature_vals = [ ]
        while d <= degree:
            feature_vals.append(acc)
            acc = acc * value
            d += 1 
        result.append(feature_vals)
    return result 