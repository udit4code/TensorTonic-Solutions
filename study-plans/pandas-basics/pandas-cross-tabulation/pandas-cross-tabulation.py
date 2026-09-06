import pandas as pd

# Cross tabulation (crosstab) computes a frequency table showing how often each combination of two categorical variables occurs. 
def cross_tab_v1(data, row_col, col_col):
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


# From first principles, using groupby, pivot and count. 
# The pipeline is essentially : 
# raw rows -> groupby(row_col, col_col) -> count each combination -> pivot into matrix -> to_dict()
def cross_tab(data, row_col, col_col):
    """
    Returns: nested dict {col_value: {row_value: frequency}}
    """
    df = pd.DataFrame(data)

    # Key idea : For every unique combination of row_col and col_col, count how many input rows have that combination, and return the result as a normal DataFrame.
    # The flow is : 
    # raw rows
    #    ↓
    # group by pair
    #    ↓
    # (row_value, col_value) -> collection of rows
    #    ↓
    # .size() [.size() counts rows in the group.]
    #    ↓
    # (row_value, col_value) -> frequency
    #    ↓
    # reset_index()
    #    ↓
    # normal 3-column DataFrame, where the 3 columns are row_col, col_col, count
    grouped = df.groupby([row_col, col_col]).size().reset_index(name="count")
    
    pivot = pd.pivot(
        grouped,
        index=row_col,
        columns=col_col,
        values="count",
    ).fillna(0).astype(int)

    return pivot.to_dict()
    