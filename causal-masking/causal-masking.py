import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Step 1 : Get Sequence Length 
    scores = np.asarray(scores, dtype=np.float64)
    T = scores.shape[-1]
    # Step 2 : Get Upper Triangular mask 
    # Why ? Token i is not allowed to see Token j if j > i
    # Say, query position is indexed by i and key position gets indexed by j 
    # Say, T = 4. Then, x = np.arange(4) = [0,1,2,3] , whose shape is (4,)
    # Now, when we do x[:, None], we can think of ":" as take all elements along that dimension 
    # and "None" as adding a new dimension there. 
    # So, after x[:, None], we take all elements along 0-th dimension and then, add a new dimension, so that shape becomes (4, 1) 
    # So, x[:, None] = [[0], [1], [2], [3]]. 
    # Similarly, when we do x[None, :], we put the new dimension before the existing dimension. So, for x[None, :], we get [[0, 1, 2, 3]] whose shape is (1, 4).
    # Now, when we do j > i, we do (1, 4) vs (4, 1) = (4, 4) via broadcasting.
    i = np.arange(T)[:, None] 
    j = np.arange(T)[None, :]
    mask = j > i 
    # Step 2 : Get masked scores 
    return np.where(mask, mask_value, scores)