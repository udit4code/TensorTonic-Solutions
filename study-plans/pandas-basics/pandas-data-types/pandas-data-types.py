import pandas as pd

def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """
    df = pd.DataFrame(data)
    num_rows, num_columns = df.shape 
    type_counts = { }
    dtypes = { }
    # df.dtypes is a series. 
    for col, dtype in df.dtypes.items():
        key = str(dtype)
        dtypes[col] = key
        if key not in type_counts:
            type_counts[key] = 1
        else:
            type_counts[key] += 1
    return {
        "dtypes" : dtypes,
        "type_counts" : type_counts,
        "num_columns" : num_columns
    }