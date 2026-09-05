import torch

# This is a capacity-planning approximation for inference memory. 
# From first principles, the GPU must hold several categories of memory at the same time:
# Total memory = model weights + KV cache + activations/workspace + runtime overhead
# Here, M = (weight_bytes + kv_cache_bytes + activation_bytes) is the base memory M. 
# The runtime overhead is approximated as a fraction of this M. 
# M_total​ = M x (1+r)
# where r might account for CUDA/PyTorch allocator overhead, temporary kernels, communication buffers, fragmentation, graph/runtime state, etc.

# Please note that min_gpu_count = math.ceil(total_required_bytes / usable_bytes_per_gpu) 
# serves valid as a lower-bound-ish memory estimate if the memory can actually be distributed across GPUs. 
# For example, with tensor/pipeline parallelism, weights can be sharded. But some objects may be replicated, and KV cache distribution depends on the serving architecture. 

# The mental model should be : 
# Required inference memory
# ├── Weights               parameter_count × dtype_size
# ├── KV cache              grows with batch × context × layers × KV heads
# ├── Activations/workspace temporary inference tensors
# └── Safety margin         allocator + kernels + fragmentation + runtime

def estimate_inference_gpu_count(
    parameter_count: int,
    bytes_per_parameter: float,
    kv_cache_bytes: int,
    activation_bytes: int,
    runtime_overhead_fraction: float,
    usable_bytes_per_gpu: int,
) -> torch.Tensor:
    """
    Returns: torch.int64 tensor of shape (2,): [total_required_bytes, min_gpu_count]
    """
    weight_bytes = parameter_count * bytes_per_parameter
    M = (weight_bytes + kv_cache_bytes + activation_bytes)
    
    total_required_bytes = math.ceil( M * (1 + runtime_overhead_fraction))
    min_gpu_count = math.ceil(total_required_bytes / usable_bytes_per_gpu)

    return torch.tensor([total_required_bytes, min_gpu_count], dtype=torch.int64)
