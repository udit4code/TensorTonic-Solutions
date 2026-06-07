import numpy as np

def impute_via_mean(X):
    X = np.array(X, dtype=np.float64)
    # Step 1 : Find column-wise mean for non_NaN values
    nan_mask = np.isnan(X)
    non_nan_mask = ~nan_mask
    # np.nansum() computes the sum while treating NaN values as if they were 0.
    col_sum = np.nansum(X, axis=0)
    col_count = non_nan_mask.sum(axis=0)
    # Avoid division-by-zero warnings
    col_mean = np.divide(
        col_sum,
        col_count,
        out=np.zeros_like(col_sum),
        where=col_count != 0
    )
    # Step 2 : Do the imputation via col_mean
    nan_cols = np.where(nan_mask)[1]
    X[nan_mask] = col_mean[nan_cols]
    return X

def impute_via_median(X):
    X = np.array(X, dtype=np.float64)
    # Step 1 : Find column-wise mean for non_NaN values
    nan_mask = np.isnan(X)
    # Compute the median of each column while ignoring NaNs
    col_median = np.nanmedian(X, axis=0)
    # Handle columns that are entirely NaN
    col_median = np.nan_to_num(col_median, nan=0.0)
    # Step 2 : Do the imputation via col_median
    # row_indices_of_NaNs, col_indices_of_NaNs = np.where(nan_mask)
    # For every NaN, find its column index and replace it with the corresponding column median
    nan_cols = np.where(nan_mask)[1]
    X[nan_mask] = col_median[nan_cols]
    return X
    
def impute(X, method="mean"):
    """
    Returns: 2D list with NaN values replaced using the specified method
    """
    result = None
    if method == "mean":
        result = impute_via_mean(X) 
    elif method == "median":
        result = impute_via_median(X)
    else:
        raise Exception(f"invalid impute method : {method}")
    return result