import pandas as pd

def rename_columns(data, rename_map):
    """
    Returns: dict mapping renamed column names to value lists
    """
    # Step 0 : Create a dataFrame 
    df = pd.DataFrame(data)
    # Step 1 : Rename columns
    # The rename(columns=...) method takes a dictionary mapping old names to new names. 
    # Only the columns present in the mapping are renamed; all other columns remain unchanged. 
    # This is safer than reassigning df.columns directly, which requires listing every column.
    df = df.rename(columns=lambda old_col_name : rename_map.get(old_col_name, old_col_name))

    return df.to_dict("list")