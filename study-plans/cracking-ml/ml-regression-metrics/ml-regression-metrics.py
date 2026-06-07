import numpy as np

def get_mse_regression_metric(y_true, y_pred):
    n = len(y_true)
    y_true = np.array(y_true, dtype=np.float64) 
    y_pred = np.array(y_pred, dtype=np.float64) 
    squared_distance = (y_true - y_pred) ** 2
    squared_distance_sum = np.sum(squared_distance)
    return squared_distance_sum/n
    

def get_mae_regression_metric(y_true, y_pred):
    n = len(y_true)
    y_true = np.array(y_true, dtype=np.float64) 
    y_pred = np.array(y_pred, dtype=np.float64)  
    abs_distance = np.abs(y_true - y_pred)
    abs_distance_sum = np.sum(abs_distance)
    return abs_distance_sum/n

def get_r2_regression_metric(y_true, y_pred):
    n = len(y_true)
    y_true = np.array(y_true, dtype=np.float64) 
    mean_y_true = np.mean(y_true)
    y_pred = np.array(y_pred, dtype=np.float64)  
    squared_distance = (y_true - y_pred) ** 2
    squared_distance_sum = np.sum(squared_distance)
    squared_true_mean_distance = (y_true - mean_y_true) ** 2
    squared_true_mean_distance_sum = np.sum(squared_true_mean_distance)
    if np.isclose(squared_true_mean_distance_sum, 0.0, rtol=1e-05, atol=1e-08, equal_nan=False):
        return 0.0
    return 1 - ((squared_distance_sum)/(squared_true_mean_distance_sum))
    
def regression_metrics(y_true, y_pred):
    """
    Returns: dict with keys "mse", "mae", "r2" rounded to 4 decimal places
    """
    assert len(y_pred) == len(y_true), "Lengths of y_true and y_pred do not match"
    result = {
        "mse" : get_mse_regression_metric(y_true, y_pred),
        "mae" : get_mae_regression_metric(y_true, y_pred),
        "r2" : get_r2_regression_metric(y_true, y_pred)
    }
    return result