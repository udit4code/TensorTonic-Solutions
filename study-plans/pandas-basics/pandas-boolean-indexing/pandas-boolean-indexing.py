import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    # Step 1 : Create a boolean mask where the column values exceed the threshold
    # df[column] > threshold creates a boolean Series of True/False values.
    # Under the hood, pandas leverage vectorized broadcasting. 
    # When we write df['age'] > 30, pandas broadcasts the scalar 30 across every element of the 'age' Series and performs the comparison in a single vectorized operation. 
    # For a column with n elements, this runs in O(n) time with C-level performance, 
    # whereas a Python loop would add interpreter overhead per element.
    mask = df[column] > threshold
    # Step 2 : Apply the mask to filter the df. 
    # Passing this mask to df[mask] returns only the rows in the df where the condition is True
    filtered_data = df[mask]
    
    return {
        "filtered_data" : filtered_data.to_dict("list"),
        "count" : len(filtered_data),
    }