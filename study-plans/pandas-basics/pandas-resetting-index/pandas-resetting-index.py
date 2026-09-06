import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    # Step 0 :  Create datafram from data
    df = pd.DataFrame(data)
    # Step 1 : Set column
    df = df.set_index(index_col)
    columns_before_reset = df.columns
    # Step 2 : ReSet index_col
    df = df.reset_index()
    columns_after_reset = df.columns
    return [
        columns_before_reset,
        columns_after_reset
    ]