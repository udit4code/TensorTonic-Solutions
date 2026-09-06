import pandas as pd

def multi_agg(data, group_col, value_col, funcs):
    """
    Returns: dict mapping function name to {group: value} dict
    """
    # Step 0 : Create dataFrame from data
    df = pd.DataFrame(data)
    # Step 1 : Group by group_col 
    grouped_df = df.groupby(group_col)
    # Step 2 : Apply funcs on the value_col per group
    # Passing a list of function names to .agg() applies each function to every group in a single call. 
    # The result is a DataFrame where columns are function names and the index contains group labels.
    # This pattern is more flexible than calling individual methods like .sum() and .mean()
    # separately. 
    # It scales to any combination of aggregation functions and produces a clean multi-column result.
    result = grouped_df[value_col].agg(funcs)
    return result.to_dict()