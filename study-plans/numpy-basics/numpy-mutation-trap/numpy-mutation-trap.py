import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    np_data = np.array(data, dtype=np.float64)
    filtered_row = np_data[row_idx]
    clipped_view = filtered_row.copy()
    assert id(filtered_row) != id(clipped_view) , f"filtered_row and clipped_view have same ids"
    low_mask = clipped_view < lo 
    high_mask = clipped_view > hi 
    clipped_view[low_mask] = lo 
    clipped_view[high_mask] = hi 
    return [filtered_row, clipped_view]