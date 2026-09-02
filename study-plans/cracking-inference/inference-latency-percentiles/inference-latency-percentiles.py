import torch

def latency_percentiles(latencies: torch.Tensor) -> torch.Tensor:
    """
    Returns: tensor of shape (3,) ordered [P50, P95, P99]
    """
    if latencies.numel() == 0 :
        raise  ValueError("latencies must be non-empty")

    if not torch.isfinite(latencies).all():
        raise ValueError("latencies must only contain finite values")
        
    percentiles = torch.tensor(
        [0.50, 0.95, 0.99],
        device=latencies.device,
        dtype=latencies.dtype,
    )

    return torch.quantile(latencies, percentiles)
