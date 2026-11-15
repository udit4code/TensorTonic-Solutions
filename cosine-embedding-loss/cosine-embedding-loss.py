import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    assert len(x1) == len(x2), f"x1 has len {len(x1)} and x2 has len {len(x2)}"
    l2_norm = lambda v : math.sqrt(sum([item**2 for item in v]))
    dot_product = lambda v1, v2 : sum([item_v1 * item_v2 for item_v1, item_v2 in zip(v1, v2)])

    cosine_product = dot_product(x1, x2) / (l2_norm(x1) * l2_norm(x2))

    if label == 1:
        return 1 - cosine_product
    else:
        return max(0, cosine_product - margin)