import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    values = [ ]
    if column in df.columns:
        values = df[column].tolist()
    else:
        raise Exception(f"column {column} not present in df.columns")
    length = len(values)
    return {
        "values" : values, 
        "length" : length,
    }