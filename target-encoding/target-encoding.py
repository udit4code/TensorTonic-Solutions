def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    frequency_map = { }
    target_sum_map = { }
    assert len(categories) == len(targets), "categories and targets are not of same length"
    n = len(categories)
    for category, target_val in zip(categories, targets):
        if category not in frequency_map:
            frequency_map[category] = 0
            target_sum_map[category] = 0
        frequency_map[category] += 1
        target_sum_map[category] += target_val
    result = [ ]
    for category in categories:
        category_mean = target_sum_map.get(category) / frequency_map.get(category) 
        result.append(category_mean)
    return result 