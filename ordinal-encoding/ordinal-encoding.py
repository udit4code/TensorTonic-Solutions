def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    index_map = {order_val : index for index, order_val in enumerate(ordering)}
    return [index_map.get(value) for value in values]