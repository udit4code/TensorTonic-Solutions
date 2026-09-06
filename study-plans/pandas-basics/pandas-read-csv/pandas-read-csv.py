import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df = pd.DataFrame(data)
    return {
        # Calling df.to_dict("list") converts each column into a key-value pair.
        "data" : df.to_dict("list"),
        # df.shape tuple gives (rows, cols)
        "shape" : list(df.shape),
        # df.columns.tolist() extracts column names as plain Python strings
        "columns" : df.columns.tolist(),
    }