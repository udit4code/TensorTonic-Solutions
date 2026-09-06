import pandas as pd

# The pd.melt() function transforms a wide-format DataFrame (one column per measurement) into a long-format DataFrame (one row per measurement). 
# The id_vars parameter specifies columns to keep fixed (they repeat for each melted row), while value_vars lists the columns whose headers become values in a new "variable" column and whose cell values go into a new "value" column.
def melt_dataframe(data, id_vars, value_vars):
    """
    Returns: dict with keys from id_vars plus 'variable' and 'value'
    """
    df = pd.DataFrame(data)
    melted = pd.melt(
        df, 
        id_vars=id_vars, 
        value_vars=value_vars
    )
    return melted.to_dict("list")