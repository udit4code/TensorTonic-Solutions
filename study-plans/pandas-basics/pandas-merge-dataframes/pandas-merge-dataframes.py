import pandas as pd

def merge_dataframes(left, right, on, how):
    """
    Returns: dict of column to value lists
    """
    # Step 0 : Create left and right dataFrames
    df_left = pd.DataFrame(left)
    df_right = pd.DataFrame(right)
    # Step 1 : Merge the 2 dataframes
    # The pd.merge() function combines two DataFrames based on a shared key column, similar to SQL JOINs. 
    # The how parameter controls which rows are kept: "inner" keeps only matching keys, "left" keeps all left rows, "right" keeps all right rows, and "outer" keeps everything.
    merged_df = pd.merge(
        df_left,
        df_right,
        on=on, 
        how=how 
    )

    return merged_df.to_dict("list")