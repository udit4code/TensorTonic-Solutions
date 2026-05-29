def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    frequency_map = {}
    for value in values:
        if value not in frequency_map:
            frequency_map[value] = 1
        else:
            frequency_map[value] += 1
    n = len(values)
    return [frequency_map.get(value) /n for value in values]