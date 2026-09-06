import pandas as pd

# The Transformation Pipeline : 
# raw rows
#    ↓
# group by group_cols
#    ↓
# group_key -> list of value_col values
#    ↓
# aggfunc
#    ↓
# group_key -> one aggregated value
#    ↓
# reset_index()
#    ↓
# normal DataFrame
#    ↓
# to_dict("list")
#    ↓
# column -> list of values

def multi_groupby(data, group_cols, value_col, aggfunc):
    """
    Returns: dict of lists (flat table with group columns + value column)
    """
    # Step 0 : Create DataFrame from data
    df = pd.DataFrame(data)
    # Step 1 : Apply Multi-Column GroupBy
    # Passing a list of columns to groupby() creates groups based on unique combinations of those columns. 
    # The aggregation is then applied within each combination, producing a result with a MultiIndex.
    # Key idea : "Partition the rows by group_cols, take value_col from each partition, reduce those values using aggfunc, then turn the result back into an ordinary DataFrame."
    multi_column_group_by_df = df.groupby(group_cols)[value_col].agg(aggfunc)
    # Step 2 : Flatten with reset_index()
    # The .reset_index() call converts the MultiIndex back into regular columns, making the result a flat DataFrame. 
    # This is easier to work with and serialize than hierarchical index structures.
    result = multi_column_group_by_df.reset_index()
    return result.to_dict("list")