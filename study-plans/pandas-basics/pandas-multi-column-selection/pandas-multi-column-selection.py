import pandas as pd

def select_columns(data, columns):
    """
    Returns: dict mapping selected column names to value lists
    """
    df = pd.DataFrame(data)
    selected_sub_df = df[columns]
    return selected_sub_df.to_dict("list")