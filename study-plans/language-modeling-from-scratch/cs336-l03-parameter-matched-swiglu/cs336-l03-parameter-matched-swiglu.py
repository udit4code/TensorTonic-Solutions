import torch
import torch.nn.functional as F
import math

def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """
    original_dtype = x.dtype
    
    # Determine input dimension d and maximum hidden width H
    d = x.shape[-1]
    H = w_g.shape[1]
    
    # Calculate the parameter-matched hidden width h
    # h = min(H, max(1, floor(base_params / (3 * d) + 0.5)))
    h = min(H, max(1, math.floor(base_params / (3 * d) + 0.5)))
    
    # The three selected matrices use a total of 3 * d * h parameters
    parameter_count = 3 * d * h
    
    # Convert inputs to float32 for stable computation as required
    x_f32 = x.to(torch.float32)
    w_g_f32 = w_g[:, :h].to(torch.float32)
    w_v_f32 = w_v[:, :h].to(torch.float32)
    w_o_f32 = w_o[:h, :].to(torch.float32)
    
    # Compute projections and gated activation: (SiLU(x W_g) * (x W_v)) W_o
    gate = F.silu(torch.matmul(x_f32, w_g_f32))
    value = torch.matmul(x_f32, w_v_f32)
    gated = gate * value
    output = torch.matmul(gated, w_o_f32)
    
    # Convert back to the input tensor's original dtype
    output = output.to(original_dtype)
    
    return {
        "output": output,
        "hidden_width": int(h),
        "parameter_count": int(parameter_count)
    }