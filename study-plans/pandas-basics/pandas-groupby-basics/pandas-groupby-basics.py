import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    # Step 0 : Create a DataFrame 
    df = pd.DataFrame(data)
    # Step 1 : Group by the specific column
    # The groupby() method splits the DataFrame into groups based on unique values in the group column. S
    # Selecting a value column and calling an aggregation method (sum, mean, count) applies the operation within each group and combines the results into a single Series.
    grouped_df = df.groupby(group_col)
    # Step 2 : Compute sum over the value_col 
    sum = grouped_df[value_col].sum()
    # Step 3 : Compute mean over the value_col
    mean = grouped_df[value_col].mean()
    # Step 4 : Compute count over the value_col
    count = grouped_df[value_col].count()

    # Calling .to_dict() on the resulting Series converts it to a Python dict where keys are group labels and values are the computed aggregates.
    return {
        "sum" : sum.to_dict(),
        "mean" : mean.to_dict(),
        "count" : count.to_dict(),
    }
    