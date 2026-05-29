def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    # Write code here
    sorted_values = sorted(values)
    
    rank_aggregation_map = { }
    frequency_map = { }
    for rank, value in enumerate(sorted_values):
        if value not in frequency_map:
            rank_aggregation_map[value] = 0
            frequency_map[value] = 0
        rank_aggregation_map[value] += (rank + 1) # ranks are 1-indexed
        frequency_map[value] += 1
        
    result = [ ]
    for value in values:
        total_rank = rank_aggregation_map[value]
        count = frequency_map[value]
        avg_rank = total_rank / count 
        result.append(avg_rank)
    return result 
    