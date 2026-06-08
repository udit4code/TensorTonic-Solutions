import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    if k <= 1:
        raise ValueError("k must be at least 2")
    if k > N:
        raise ValueError("k cannot exceed N")
    # Step 1 : Create indices. Eg. np.arange(N), N = 5 gives [0, 1, 2, 3, 4]
    indices = np.arange(N)
    if shuffle:
        # Step 1.1 : Shuffle the indices if requested
        if rng is None:
            rng = np.random.RandomState()
        rng.shuffle(indices)
    # Step 2 : Compute fold sizes. In K-Fold Cross Validation we want folds to be as equal in size as possible.
    # np.full(k, N//k) means to Create an array of length k, and fill every element with N // k.
    # Step 2.1 : Give every fold an equal base size. For N = 10 and k = 3, we get np.full(k, N//k) = [3, 3, 3]
    fold_sizes = np.full(k, N // k, dtype=int)
    # Step 2.2 : Distribute the leftover samples one-by-one to the first few folds.
    # In this case, N % k = 10 % 3 = 1. So, fold_sizes[0: 1] = {fold_sizes[0]}, which we increment. So, we get [3, 3, 3] -> [3 + 1, 3, 3] = [4, 3, 3]
    fold_sizes[0 : N % k] += 1
    # Step 3: Prepare splits as per fold_sizes
    splits = []
    current = 0
    for fold_size in fold_sizes:
        # start and end imply the starting index (inclusive) and ending index (non-inclusive) of the validation fold
        start = current
        end = current + fold_size
        # Validation indices for this fold.
        val_idx = indices[start:end]
        # Training indices are everything else.
        train_idx = np.concatenate([ indices[:start] , indices[end:] ])
        splits.append((train_idx, val_idx))
        current = end
    return splits

    
