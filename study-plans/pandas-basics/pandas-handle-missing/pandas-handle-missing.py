import pandas as pd

def handle_missing(data, fill_value):
    """
    Returns: dict with 'null_counts' (dict) and 'cleaned_data' (dict)
    """
    # Step 1 : Create dataframe
    df = pd.DataFrame(data)
    # Step 2 : Get null counts per column 
    null_counts = { }
    for key, value in df.isnull().sum().items():
        null_counts[key] = int(value)
    # Step 3 : Fill null values with fill_value
    cleaned_df = df.fillna(fill_value)
    return {
        "null_counts" : null_counts,
        "cleaned_data" : cleaned_df.to_dict("list"),
    }