def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    if n_items <= 0:
        return 0.0
        
    # Step 1: Get unique items across all recommendations using .update()
    recommended_set = set()
    for recommendation in recommendations:
        recommended_set.update(recommendation)
        
    # Step 2: Compute catalog coverage
    coverage = len(recommended_set) / n_items
    return float(coverage)