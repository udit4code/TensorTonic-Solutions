import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    pairs = [ ]
    n = token_ids.shape[0]
    for center in range(n):
        low = max(0, center - window)
        high = min(center + window, n - 1)
        for index in range(low, high + 1, 1):
            if index != center:
                pairs.append(
                    [
                        int(token_ids[center].item()), 
                        int(token_ids[index].item())
                    ]
                )
    if len(pairs) <= 0:
        return torch.zeros((0, 2), dtype=token_ids.dtype)
    return torch.tensor(pairs, dtype=token_ids.dtype)
    
