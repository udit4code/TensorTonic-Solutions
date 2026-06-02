def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    hash_map = {}
    order = [] # Maintains the order in which the records entered.
    for record in records:
        hash_key = tuple([record[column] for column in key_columns])
        if hash_key not in hash_map:
            hash_map[hash_key] = [ ]
            order.append(hash_key)
        hash_map[hash_key].append(record)
    result = [ ]
    for hash_key in order: 
        hash_values = hash_map[hash_key]
        if strategy == "first":
            result.append(hash_values[0])
        elif strategy == "last":
            result.append(hash_values[-1])
        elif strategy == "most_complete":
            result.append(min(hash_values, key=lambda record: sum(1 for v in record.values() if v is None)))
    return result