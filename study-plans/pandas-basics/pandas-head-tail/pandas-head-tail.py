import pandas as pd

def head_tail(data, n):
    """
    Returns: dict with 'head' and 'tail' (both dicts of column -> list)
    """
    df = pd.DataFrame(data)
    # The head(n) and tail(n) let us peek at the beginning and end of df. 
    head = df.head(n).to_dict("list")
    tail = df.tail(n).to_dict("list")
    return {
        "head" : head, 
        "tail" : tail, 
    }