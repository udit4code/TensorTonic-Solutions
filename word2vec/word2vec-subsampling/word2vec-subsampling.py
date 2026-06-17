import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # Get total frequency
    N = counts.shape # No of elements of count vector, which is basically the vocabulary size
    total_frequency = torch.sum(counts)
    frequency_vector = counts  / total_frequency
    threshold = torch.tensor(t, dtype=torch.float32)
    # Get the probability vector
    ones = torch.ones(N)
    probability_vector = torch.min(ones, torch.sqrt(t / frequency_vector))
    return probability_vector
    
