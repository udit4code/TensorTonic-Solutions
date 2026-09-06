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
    # Collect the number of duplicate rows
    # The duplicated() method returns a boolean Series marking duplicate rows. 
    # By default, keep=first, which means that the first occurrence is not a duplicate, while the subequent ones are. 
    duplicate_row_count = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    rows_after = len(df)

    assert rows_before - rows_after == duplicate_row_count
    
    return [
        rows_before, 
        rows_after, 
        df.to_dict("list")
    ]