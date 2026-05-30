import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    # Step 1 : Create a np array of dtype float64
    np_data = np.array(data, dtype=np.float64)
    # Step 2 : Filter the row based on row_idx
    filtered_row = np_data[row_idx]
    # Step 3 : Create a view out of filtered_row, which can be modified separately.
    clipped_view = filtered_row.copy()
    assert id(filtered_row) != id(clipped_view) , f"filtered_row and clipped_view have same ids"
    # Step 4 : Get the mask where value is less than lo and higher than hi 
    low_mask = clipped_view < lo 
    high_mask = clipped_view > hi 
    # Step 5 : Clip
    clipped_view[low_mask] = lo 
    clipped_view[high_mask] = hi 
    return [filtered_row, clipped_view]