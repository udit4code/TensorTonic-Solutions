import pandas as pd

# The pivot_table() function reshapes data from long format into a summary table. 
# Rows are grouped by the index column, columns are spread by the columns column, and cells contain the aggregated values. 
# This is one of the most powerful tools for data summarization.
def create_pivot(data, index, columns, values, aggfunc):
    """
    Returns: nested dict {column_value: {index_value: agg_result}}
    """
    # Step 0 : Create a DataFram from data
    df = pd.DataFrame(data)
    # Step 1 :  Create pivot-table 
    pivot_df = pd.pivot_table(
        df,
        values=values, # columns to aggregate
        index=index, # row labels
        columns=columns, # column labels
        aggfunc=aggfunc,
        # Setting fill_value=0 replaces NaN for combinations that don't exist in the data. Without it, missing combinations would appear as NaN, which complicates further analysis.
        fill_value=0
    )
    return pivot_df.to_dict()
    