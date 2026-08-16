import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Step 1 : Find the pid of the current program instance
    pid = tl.program_id(0)

     # Step 2 : Find the offsets (addresses) of the input data (given by the pointers) on which the current program instance will operate 
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

     # Step 3 : This is for the edge case, when one of the program instances may end up working on less than BLOCK_SIZE elements. 
    # So, to filter them out, we need to create a mask before the next step. 
    # We don't want to hit the out-of-bound error due to invalid memory access. 
    mask = offsets < n 

    # Step 4 : Given the offsets for the current program instance, we now load the data from global memory HBM to registers within the GPU core, since we have got the addresses given via offsets. 
    # This is a pure-map with one read, one compute and one write. 
    x_vals = tl.load(x_ptr + offsets, mask=mask) # Read (via load)

    # Triton exposes an elementwise maximum that fuses the clamp into one instruction. 
    # A branch-free select against the scalar zero works just as well.
    out_vals = tl.where(x_vals > 0, x_vals, 0.0) # Compute Element-wise max(0, x) 

    # Step 5 : Write back the result from the register to the HBM for the current thread/program instance.
    tl.store(out_ptr + offsets, out_vals, mask=mask) # Write
    


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch relu_kernel: out = max(x, 0)."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)