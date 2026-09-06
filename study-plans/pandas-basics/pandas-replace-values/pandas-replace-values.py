import pandas as pd

def replace_values(data, column, old_val, new_val):
    """
    Returns: dict with 'data' (dict) and 'count' (int)
    """
    # Step 0 : Create DataFrame 
    df = pd.DataFrame(data)
    # Step 1 : Count number of rows where replacements will happen. 
    # (df[column] == old_val) gives a boolean mask and .sum() counts True values in that mask. 
    count = (df[column] == old_val).sum()
    # Step 2 : Do the replacement
    # The .replace() method on a Series swaps every occurrence of the old value with the new one. It works with both numeric and string values.
    df[column] = df[column].replace(old_val, new_val)
    return {
        "data" : df.to_dict("list"),
        "count" : count,
    }