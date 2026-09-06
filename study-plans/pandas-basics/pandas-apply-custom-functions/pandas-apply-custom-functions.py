import pandas as pd

def apply_transform(data, column, operation):
    """
    Returns: dict with original columns plus column_transformed
    """
    df = pd.DataFrame(data)
    if operation == "normalize":
        col_min = df[column].min()
        col_max = df[column].max()
        df[column + "_transformed"] = df[column].apply(
            lambda x: round((x - col_min) / (col_max - col_min), 4)
        )
    elif operation == "rank":
        df[column + "_transformed"] = df[column].rank().astype(int)
    elif operation == "cumsum":
        df[column + "_transformed"] = df[column].cumsum()
    elif operation == "double":
        df[column + "_transformed"] = df[column].apply(lambda x: x * 2)
    return df.to_dict("list")