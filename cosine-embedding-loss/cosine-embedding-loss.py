import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Computes the Cosine Embedding Loss between two embedding vectors x1 and x2.
    Returns the loss as a float.
    
    Args:
        x1: First input vector (list of numbers).
        x2: Second input vector (list of numbers).
        label: Target indicator (typically 1 for similar pairs, -1 for dissimilar pairs).
        margin: Threshold margin used to penalize dissimilar pairs if they are too close.
    """
    # Step 1: Dimension Check
    # Intuition: Both vectors must have the same feature dimension (length d) 
    # to perform element-wise operations and compute an angle/similarity.
    assert len(x1) == len(x2), f"x1 has len {len(x1)} and x2 has len {len(x2)}"
    
    # Helper functions for vector math:
    # l2_norm: Computes the Euclidean magnitude (length) of a vector: ||v|| = sqrt(sum(v_i^2))
    l2_norm = lambda v : math.sqrt(sum([item**2 for item in v]))
    
    # dot_product: Computes the inner product between two vectors: v1 · v2 = sum(v1_i * v2_i)
    dot_product = lambda v1, v2 : sum([item_v1 * item_v2 for item_v1, item_v2 in zip(v1, v2)])

    # Step 2: Compute Cosine Similarity
    # Formula: cos(theta) = (x1 · x2) / (||x1|| * ||x2||)
    # Intuition: Measures the angle between two vectors, yielding a value between -1 and 1.
    # - 1.0 means vectors point in the exact same direction (identical/similar).
    # - 0.0 means they are orthogonal (independent).
    # - -1.0 means they point in opposite directions.
    cosine_product = dot_product(x1, x2) / (l2_norm(x1) * l2_norm(x2))

    # Step 3: Compute Loss based on the Target Label
    if label == 1:
        # Case A: Similar Pairs (label = 1)
        # Intuition: We want the cosine similarity to be maximized (approaching 1.0).
        # Loss = 1 - similarity. 
        # - If similarity = 1.0 (perfect match), loss = 0.
        # - If similarity = -1.0 (opposite), loss = 2.0 (maximum penalty).
        return 1 - cosine_product
    else:
        # Case B: Dissimilar Pairs (label = -1)
        # Intuition: We want the similarity to stay below a specified 'margin'.
        # - If similarity is less than margin, loss is 0 (no penalty).
        # - If similarity exceeds the margin, we penalize the model proportional 
        #   to how much it crossed the boundary (cosine_product - margin).
        return max(0, cosine_product - margin)