import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    np_data = np.array(data, dtype=np.float64)
    # Step 1 : Get the mask matrix
    boolean_mask = np.zeros(np_data.shape, dtype=np.float64)
    boolean_mask[np_data > threshold] = 1.0
    # Step 2 : 
    # np.any(np_data > threshold, axis=1) collapses each row into a single bool: True if at least one element passed. It returns  any_mask of shape (m,)
    any_mask = np.any(np_data > threshold, axis=1)
    # Since any_mask is of shape (m, ) , we need to convert it into (m, 1)
    any_mask_2d = np.reshape(any_mask, (-1, 1))
    # np.where(mask_2d, a, 0.0) then selects the original row value where the mask is True or fills with 0.0 where False.
    any_filtered = np.where(any_mask_2d, np_data, 0.0)
    # Step 3 : 
    # np.all(a > threshold, axis=1) requires every element to pass. It returns  any_mask of shape (m,)
    all_mask = np.all(np_data > threshold, axis=1)
    # Since all_mask is of shape (m, ) , we need to convert it into (m, 1)
    all_mask_2d = np.reshape(all_mask, (-1, 1))
    all_filtered = np.where(all_mask_2d, np_data, 0.0)
    return np.stack([boolean_mask, any_filtered, all_filtered])