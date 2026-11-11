import torch 

def topk_accuracy(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals, using a simple for loop.
    """
    total = len(targets)
    if total == 0:
        return 0.0
    
    correct_count = 0
    
    # Iterate through each sample's logits and its corresponding target label
    for sample_logits, target in zip(logits, targets):
        # Sort the class indices based on their logit scores in descending order
        sorted_indices = sorted(range(len(sample_logits)), key=lambda i: sample_logits[i], reverse=True)
        
        # Select the top-k predicted class indices
        top_k_indices = sorted_indices[:k]
        
        # Check if the true target is present in the top-k predictions
        if target in top_k_indices:
            correct_count += 1
            
    # Compute accuracy ratio and round to 4 decimal places
    return round(correct_count / total, 4)
def topk_accuracy_naive(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals, using a simple for loop.
    Breaks ties by lower class index, including ties at the cutoff.

    Example Walkthrough:
        Input: logits = [[1, 2, 3], [0, 5, 0]], targets = [2, 1], k = 1
        Output: 1.0
        
        Sample 0:
          - logits: [1, 2, 3] for classes 0, 1, and 2
          - sorted indices: [2, 1, 0] (scores are 3, 2, 1)
          - top_1 indices (k=1): [2]
          - target: 2 -> Present in top_1! (correct_count becomes 1)
          
        Sample 1:
          - logits: [0, 5, 0] for classes 0, 1, and 2
          - sorted indices: [1, 0, 2] (scores are 5, 0, 0; class 0 breaks tie with class 2 via lower index)
          - top_1 indices (k=1): [1]
          - target: 1 -> Present in top_1! (correct_count becomes 2)
          
        Final Calculation:
          - correct_count / total = 2 / 2 = 1.0
    """
    total = len(targets)
    if total == 0:
        return 0.0
    
    correct_count = 0
    
    # Step 1 : Iterate through each sample's logits and its corresponding target label
    for sample_logits, target in zip(logits, targets):
        # Step 1.1 : Sort indices by logit score (descending) and then by class index (ascending).
        # Using a tuple key (logit, -index) with reverse=True ensures that:
        # 1. Higher logit values come first.
        # 2. For identical logits, a higher negative index (which corresponds to a lower class index) comes first.
        sorted_indices = sorted(
            range(len(sample_logits)), 
            key=lambda i: (sample_logits[i], -i), 
            reverse=True
        )
        
        # Step 1.2 : Select the top-k predicted class indices (ties at the cutoff are naturally resolved 
        # in favor of the lower class index due to the deterministic secondary sort key)
        top_k_indices = sorted_indices[:k]
        
        # Step 1.3 : Check if the true target is present in the top-k predictions
        if target in top_k_indices:
            correct_count += 1
            
    # Compute accuracy ratio and round to 4 decimal places
    return round(correct_count / total, 4)