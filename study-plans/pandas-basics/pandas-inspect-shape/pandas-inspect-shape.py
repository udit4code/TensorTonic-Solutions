import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df = pd.DataFrame(data)
    rows, cols = df.shape
    return {
        "rows" : rows,
        "cols" : cols, 
        "columns" : df.columns.tolist(),
        # The df.dtypes Series maps each column to its pandas dtype.
        "dtypes" : {col : str(dtype) for col, dtype in df.dtypes.items()},
        "total_values" : int(df.size),
    }