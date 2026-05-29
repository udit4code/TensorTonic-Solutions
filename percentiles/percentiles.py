import numpy as np

def get_percentile_per_query(x, query):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    # sort values
    x = np.sort(x)
    n = len(x)
    if n == 0:
        raise ValueError("empty array")
    # percentile position
    pos = (query / 100) * (n - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    # exact index
    if lower == upper:
        return x[lower]
    # interpolation weight
    weight = pos - lower
    return (x[lower] * (1 - weight) + x[upper] * weight)


def percentiles(x, queries):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    # return np.percentile(x, q, method="linear")
    result = list(map(lambda q: get_percentile_per_query(x, q), queries))
    return np.array(result)