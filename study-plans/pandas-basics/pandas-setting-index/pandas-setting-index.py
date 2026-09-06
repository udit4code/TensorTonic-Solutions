import pandas as pd

def set_index_column(data, index_col):
    """
    Returns: dict with 'index_values', 'columns', 'data'
    """
    df = pd.DataFrame(data)
    # Step 1 : Set the specified column as the DataFrame index 
    # The set_index() method moves a column from the data into the index. 
    # The column is no longer listed among regular columns, and its values become row labels.
    df = df.set_index(index_col)
    # Step 2 : Extract index values as a list
    index_values = df.index.tolist()
    # Step 3 : Extract the remaining column names and their data
    columns = df.columns
    return {
        "index_values" : index_values,
        "columns" : columns, 
        "data" : df.to_dict("list")
    }