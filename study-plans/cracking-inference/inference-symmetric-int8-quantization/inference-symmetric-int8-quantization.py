import torch
from typing import Tuple

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (quantized int8 tensor, scale scalar tensor, dequantized float tensor)
    """
    # Step 1: Find largest absolute value
    max_abs = torch.max(torch.abs(x))
    # Step 2: Compute scale
    scale = max_abs / 127.0

    # Handle all-zero tensor
    if max_abs == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    # Step 3: Map float values into [-127, 127]
    quantized = torch.round(x / scale)
    # Step 4: Clamp to valid symmetric INT8 range
    quantized = torch.clamp(quantized, -127, 127)
    # Step 5: Store as actual int8
    quantized = quantized.to(torch.int8)
    # Step 6: Approximate original values
    dequantized = quantized.to(x.dtype) * scale

    return quantized, scale, dequantized
