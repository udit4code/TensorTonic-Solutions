import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    # Step 1 : Create DataFrame
    df = pd.DataFrame(data)
    rows_before = len(df)
    # Step 2 : Drop duplicates from df. 
    # The drop_duplicates() method checks all columns by default and removes rows where every value matches a previously seen row. 
    # It keeps the first occurrence by default.
    df = df.drop_duplicates(keep="first")
    rows_after = len(df)
    
    return [
        rows_before, 
        rows_after, 
        df.to_dict("list")
    ]