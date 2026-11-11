import torch

def topk_accuracy(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals, using torch.argsort.
    Breaks ties by lower class index, including ties at the cutoff.
    """
    logits = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    
    N, num_classes = logits.shape  
    
    # Step 0 : Ensure k does not exceed the total number of available classes
    k = min(k, num_classes)
    
    # Step 1 : Use torch.argsort in descending order with stable=True.
    # Because columns are ordered from class index 0 to num_classes - 1,
    # a stable sort preserves the original column order for equal logit values,
    # automatically breaking ties in favor of the lower class index.
    sorted_indices = torch.argsort(logits, dim=1, descending=True, stable=True)
    
    # Step 2 : Select the top-k predicted class indices for each sample
    topk_preds = sorted_indices[:, :k]
    
    # Step 3 : Reshape targets to (N, 1) for broadcast comparison against top-k predictions
    # Why ? targets is initially shaped as (N,)
    targets_reshaped = targets.view(-1, 1)
    
    # Step 4 : Check if the ground truth target is present anywhere within the top-k predictions
    correct = (topk_preds == targets_reshaped).any(dim=1).sum().item()
    
    # Step 5 : Compute accuracy ratio and round to 4 decimal places
    accuracy = correct / N
    return round(float(accuracy), 4)

    
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