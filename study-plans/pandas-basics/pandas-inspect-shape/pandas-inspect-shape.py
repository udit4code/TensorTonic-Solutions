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
        # The df.columns attribute returns an Index object containing the column labels. 
        # The Index behaves like an immutable array. 
        # We can iterate over it, check membership with in, and convert it to a list with .tolist()
        "columns" : df.columns.tolist(),
        # The df.dtypes Series maps each column to its pandas dtype.
        "dtypes" : {col : str(dtype) for col, dtype in df.dtypes.items()},
        "total_values" : int(df.size),
    }