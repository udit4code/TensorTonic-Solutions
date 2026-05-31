import numpy as np

def tile_diff(data, reps):
    """Returns: np.ndarray of shape (2, m*reps, n), stacked tiled array and padded differences"""
    data = np.array(data, dtype=np.float64)
    # np.tile(A, (r, c))
    # r = number of vertical repetitions
    # c = number of horizontal repetitions
    # Example:
    # A = [[1,2], [3,4]]
    # np.tile(A, (2,1)) becomes: [[1,2], [3,4], [1,2], [3,4]]
    tiled = np.tile(data, (reps, 1))
    # Compute consecutive row differences.
    # np.diff(A, axis=0) calculates:
    # 1. row[1] - row[0]
    # 2. row[2] - row[1]
    # 3. row[3] - row[2]
    # Example: [[1,2], [3,4], [1,2], [3,4]] becomes: [[ 2, 2], [-2,-2], [ 2, 2]]
    # Shape decreases by 1 along axis=0.
    diff = np.diff(tiled, axis=0)
    # np.diff reduces the number of rows by 1.
    # Original tiled shape: (4,2)
    # diff shape: (3,2)
    # Add one zero row at the bottom so that diff and tiled have the same shape.
    # ((0,1),(0,0)) means:
    # rows: 0 on top and 1 on bottom
    # cols: 0 on left and 0 on right
    diff_padded = np.pad(diff, ((0, 1), (0, 0)))
    return np.stack([tiled, diff_padded])