import pandas as pd

def iloc_selection(data, row, col):
    """
    Returns: list [element, row_values, col_values]
    """
    df = pd.DataFrame(data)
    # The iloc accessor uses integer positions to select data. 
    # iloc[row, col] gets a single element. 
    element = df.iloc[row, col]
    # iloc[row, :] gets an entire row.
    row_values = df.iloc[row, :].tolist()
    # iloc[:, col] gets an entire column. 
    # The colon : means "all positions along that axis."
    col_values = df.iloc[:, col].tolist()
    return [
        element, 
        row_values,
        col_values
    ]