import pandas as pd

# Cross tabulation (crosstab) computes a frequency table showing how often each combination of two categorical variables occurs. 
def cross_tab(data, row_col, col_col):
    """
    Returns: nested dict {col_value: {row_value: frequency}}
    """
    # Step 0 : Create DataFrame from data
    df = pd.DataFrame(data)
    # Step 1 : Compute a cross-tabulation of 2 specified columns
    # The pd.crosstab() function computes a frequency table showing how often each pair of values appears together. 
    # It is equivalent to groupby + count + pivot, but in a single call. 
    cross_table_df = pd.crosstab(df[row_col], df[col_col])
    return cross_table_df.to_dict()
    