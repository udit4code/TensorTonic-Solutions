import pandas as pd

# The .astype() method converts a Series to the specified type.
def change_dtype(data, column, target_type):
    """
    Returns: list [dtypes_before, dtypes_after] (both dicts)
    """
    # Step 0 : Create dataFrame out of data
    df = pd.DataFrame(data)
    # Step 1 : Record data types before
    dtypes_before = df.dtypes.astype(str).to_dict()
    # Step 2 : Switch datatype
    df[column] = df[column].astype(target_type)
    # Step 3 : Record data types after
    dtypes_after = df.dtypes.astype(str).to_dict()
    return [
        dtypes_before,
        dtypes_after
    ]