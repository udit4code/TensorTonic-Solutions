import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
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
    # Step 6 : Atomically add block sum to out
    tl.atomic_add(out_ptr, block_sum)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)