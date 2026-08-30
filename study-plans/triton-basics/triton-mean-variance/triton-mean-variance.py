import torch
import triton
import triton.language as tl


# Key idea : We use 2 scratch-buffers (initialized to zero and pointed to by sum_ptr and sumsq_ptr) and 2 atomic-add operations. For each program instance, the combine step is independent and hence, we can use tl.atomic_add to do the reduction on the global scratch_buffers. 

@triton.jit
def mean_var_kernel(x_ptr, sum_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Step 1 : Find the pid of the current program instance
    pid = tl.program_id(0)
    # Step 2 : Create logical offsets, starting from block_start_offset
    block_start_offset = pid * BLOCK_SIZE
    offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
    # Step 3 : Create Mask (We want invalid positions, i.e positions outside the block, to behave as zero) 
    mask = offsets < n   

    # Step 4 : Given the offsets for the current program instance, we now load the data from global memory HBM to registers within the GPU core, since we have got the addresses given via offsets.
    x_vals = tl.load(x_ptr + offsets, mask=mask)

    # Step 5 : Obtain the block sum from the current program
    block_sum = tl.sum(x_vals, axis=0)

    # Step 6 : Obtain the block square_sum from the current program 
    block_squared_sum = tl.sum(x_vals * x_vals, axis=0)

    # Step 7 : Now, do atomic add for the block's aggregated statistics 
    tl.atomic_add(sum_ptr, block_sum)
    tl.atomic_add(sumsq_ptr, block_squared_sum)
    

# Assuming x, mean_out and var_out are CUDA tensors, under the hood : 
# 1. n = x.numel() runs on CPU, where it reads the tensor metadata. 
# 2. torch.zeros(1, device='cuda', dtype=torch.float32) is called by CPU, but 
# memory is allocated on GPU memory. Zero-Initialisation is done for a device memory, via a CUDA operation 
# The kernel launch via mean_var_kernel[grid](x, sum_buf, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)  is initiated by the CPU, and the kernel body runs on GPU. 
# mean = sum_buf / n is called by CPU, but computation runs on the GPU because sum_buf is a CUDA tensor . 
# CPU initiates mean_out.copy_(mean) . If mean_out is CUDA, this is device-to-device GPU copying.

# KEY TAKEAWAY : Python Code runs on the CPU. Operations involving CUDA tensors are scheduled for the GPU. 
# The CPU usually does not wait for the output of every GPU operation. 
# CPU places operations into a CUDA stream in a given order : zero-buffers -> run mean_var_kernel -> compute mean -> compute variance -> copy outputs. 
# CUDA preserves this ordering within the same stream. 

    
def solve(x: torch.Tensor, mean_out: torch.Tensor, var_out: torch.Tensor) -> None:
    """Launch mean_var_kernel and finalize mean and variance."""
    n = x.numel()
    # The Python call happens on the host, but the buffers themselves live in GPU memory. We pre-zero the sum_buf and sumsq_buf . 
    sum_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mean_var_kernel[grid](x, sum_buf, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE) 
    
    mean = sum_buf / n
    var = sumsq_buf / n - mean * mean
    mean_out.copy_(mean)
    var_out.copy_(var)