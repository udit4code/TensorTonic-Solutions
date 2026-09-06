import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    frames = [pd.DataFrame(d) for d in dfs]
    # The pd.concat() function takes a list of DataFrames and stacks them along axis 0 (rows) by default. 
    # Columns are aligned by name: if both DataFrames have a column called "age", those values line up. 
    # Columns that exist in only one DataFrame get NaN in the rows from the other.
    # Without ignore_index=True, each DataFrame keeps its original index (e.g., both start at 0), producing duplicate index values. Setting it to True creates a fresh 0-based index for the combined result, which is almost always what you want when appending rows from different sources.
    result = pd.concat(frames, ignore_index=True)
    return [
        list(result.shape), 
        result.to_dict("list")
    ]