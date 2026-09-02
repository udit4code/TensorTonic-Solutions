import torch


def quantile_from_scratch(
    x: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    x: 1D tensor
    q: tensor of quantiles in [0, 1]

    Returns:
        tensor with one value per quantile
    """
    x_sorted, _ = torch.sort(x)

    n = x_sorted.numel()

    # Fractional index into sorted data
    positions = q * (n - 1)

    # Neighboring integer indices
    lower_idx = torch.floor(positions).long()
    upper_idx = torch.ceil(positions).long()

    # How far are we between lower and upper?
    weight = positions - lower_idx.to(positions.dtype)

    lower_values = x_sorted[lower_idx]
    upper_values = x_sorted[upper_idx]

    # Linear interpolation
    return lower_values + weight * (upper_values - lower_values)

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

    # return torch.quantile(latencies, percentiles)
    return quantile_from_scratch(latencies, percentiles)
