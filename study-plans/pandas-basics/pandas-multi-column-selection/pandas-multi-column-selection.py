import pandas as pd

def select_columns(data, columns):
    """
    Returns: dict mapping selected column names to value lists
    """
    df = pd.DataFrame(data)
    # The most direct way to select multiple columns is to pass a list of column names inside brackets: subset = df[['name', 'age', 'salary']] 
    # This always returns a DataFrame, even if the list contains only one name. The columns appear in the order specified in the list.
    selected_sub_df = df[columns]
    return selected_sub_df.to_dict("list")