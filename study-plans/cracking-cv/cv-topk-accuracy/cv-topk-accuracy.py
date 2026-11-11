def topk_accuracy(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals, using a simple for loop.
    Breaks ties by lower class index, including ties at the cutoff.
    """
    total = len(targets)
    if total == 0:
        return 0.0
    
    correct_count = 0
    
    # Iterate through each sample's logits and its corresponding target label
    for sample_logits, target in zip(logits, targets):
        # Sort indices by logit score (descending) and then by class index (ascending).
        # Using a tuple key (logit, -index) with reverse=True ensures that:
        # 1. Higher logit values come first.
        # 2. For identical logits, a higher negative index (which corresponds to a lower class index) comes first.
        sorted_indices = sorted(
            range(len(sample_logits)), 
            key=lambda i: (sample_logits[i], -i), 
            reverse=True
        )
        
        # Select the top-k predicted class indices (ties at the cutoff are naturally resolved 
        # in favor of the lower class index due to the deterministic secondary sort key)
        top_k_indices = sorted_indices[:k]
        
        # Check if the true target is present in the top-k predictions
        if target in top_k_indices:
            correct_count += 1
            
    # Compute accuracy ratio and round to 4 decimal places
    return round(correct_count / total, 4)