import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
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

    # We pre-multiply and hard-code inv_sqrt2, so that it gets stored on register of the thread. 
    # Dividing by sqrt(2) requires a reciprocal instruction. 
    # Pre-multiplying by the constant 1 / sqrt(2) ≈ 0.7071067811865475 is one FMA-friendly multiply. 
    # The numerical difference is negligible at float32 and the kernel stays uniformly fast across all tiles.
    inv_sqrt2 = 0.7071067811865475

    # Step 5 : Triton exposes the error function as tl.math.erf . 
    # We apply it elementwise on the loaded tile. 
    out_vals = 0.5 * x_vals * (1.0 + tl.math.erf(x_vals * inv_sqrt2))

    # Step 5 : Write back the result from the register to the HBM for the current thread/program instance.
    tl.store(out_ptr + offsets, out_vals, mask=mask) # Write


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gelu_kernel: out = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)