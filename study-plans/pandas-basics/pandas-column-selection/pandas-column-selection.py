import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    values = [ ]
    # Why This Solution Works ? 
    # Column Access by Name Using df[column]. 
    # df[column] selects a single column as a pandas Series. 
    # This is the most common way to access columns in pandas. 
    # The .tolist() method converts the Series to a plain Python list.
    if column in df.columns:
        values = df[column].tolist()
    else:
        raise Exception(f"column {column} not present in df.columns")
    length = len(values)
    return {
        "values" : values, 
        "length" : length,
    }