import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # KEY ASSUMPTION : This is a single program instance design.
    # It hinges on the assumption that the entire vector pointed at by x_ptr fits into one Triton block. So, we don't need pid to compute offsets.  

    # Step 1 : Prepare offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Step 2 : Load the data
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # Step 3 : Take the max from the loaded x_vals
    block_max = tl.max(x, axis=0)

    # Step 4 : Write the result back to the location pointed by out_ptr 
    tl.store(out_ptr, block_max)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    # This kernel launches exactly one program instance, which processes the entire vector, so its program id is always 0. 
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


