def edit_distance_v1(s1, s2):
    cache = {}
    # We want to convert s1 to s2
    def compute_util(s1, s2, idx_1, idx_2):
        nonlocal cache
        if idx_1 < 0:
            return idx_2 + 1
        if idx_2 < 0:
            return idx_1 + 1
        key = (idx_1, idx_2)
        if key not in cache:
            if s1[idx_1] == s2[idx_2]:
                cache[key] = compute_util(s1, s2, idx_1 - 1, idx_2 - 1)
            else:
                insert = compute_util(s1, s2, idx_1, idx_2 - 1)
                delete = compute_util(s1, s2, idx_1 - 1, idx_2)
                replace = compute_util(s1, s2, idx_1 - 1, idx_2 - 1)
                # Cost of each operation is 1
                cache[key] = min(replace, insert, delete) + 1
        return cache.get((idx_1, idx_2))
    return compute_util(s1, s2, len(s1) - 1, len(s2) - 1)
    
def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # Write code here
    return edit_distance_v1(s1, s2)
    