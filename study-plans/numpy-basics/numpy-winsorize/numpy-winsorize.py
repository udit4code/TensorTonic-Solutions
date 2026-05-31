import numpy as np

# Winsorization means : Replace values below a lower percentile with the lower percentile value and Replace values above an upper percentile with the upper percentile value.

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    # Assume shape of np_data is (m, n) = (2, 2)
    # np_data = [ [1., 2.],[3., 4.]]
    np_data = np.array(data, dtype=np.float64)
    # Assume shape of low and high is (n, ) = (2, ) and lo_q = 2 and hi_q = 3
    # Now, we calculate 2nd percentile of each column.
    # Compute the lo_q percentile for each column.
    # Column 0: [1,3] -> 2nd percentile = 1.04
    # Column 1: [2,4] -> 2nd percentile = 2.04
    # Result: low = [1.04, 2.04]
    low = np.percentile(np_data, lo_q, axis=0)
    # Compute the hi_q percentile for each column.
    # Column 0: [1,3] -> 3rd percentile = 1.06
    # Column 1: [2,4] -> 3rd percentile = 2.06
    # Result: high = [1.06, 2.06]
    high = np.percentile(np_data, hi_q, axis=0)
    # Clipping is where broadcasting happens, np_data has shape (2, 2) and low, high have shape (2, )
    # Now, via broadcasting, (2, 2) vs (, 2) = (2, 2) vs (1, 2) = (2, 2)
    # So, low = [1.04, 2.04] -> [[1.04, 2.04], [1.04, 2.04]]
    # and high = [1.06, 2.06] -> [[1.06, 2.06], [1.06, 2.06]]

    # Clip every element column-wise. Values below low are replaced by low.
    # Values above high are replaced by high.
    # low and high have shape (n,) and So NumPy broadcasts them to shape (m,n).
    # clipped = [[1.04, 2.04], [1.06, 2.06]]
    clipped = np.clip(np_data, low, high)

    # Identify values below the lower percentile.
    # low broadcasts from shape (n,) to shape (m,n).
    # Result: [[ True,  True], [False, False]]
    low_mask = np_data < low 
    high_mask = np_data > high 

    # Identify values above the upper percentile.
    # Result: [[False, False], [ True,  True]]
    
    result = np.stack([clipped, low_mask, high_mask])
    return result