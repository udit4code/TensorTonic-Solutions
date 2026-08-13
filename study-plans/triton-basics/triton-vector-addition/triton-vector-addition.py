import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Step 1 : Find the pid of the current program instance
    pid = tl.program_id(0)

    # Step 2 : Find the offsets (addresses) of the input data (given by the pointers) on which the current program instance will operate 
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Step 3 : This is for the edge case, when one of the program instances may end up working on less than BLOCK_SIZE elements. 
    # So, to filter them out, we need to create a mask before the next step. 
    # We don't want to hit the out-of-bound error due to invalid memory access. 
    mask = offsets < n 

    # Step 4 : Given the offsets for the current program instance, we now load the data from global memory HBM to registers within the GPU core, since we have got the addresses given via offsets.
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    y_vals = tl.load(y_ptr + offsets, mask=mask)

    # Step 5 : Do the Addition computation in 1 shot on the GPU register. 
    # Registers are private per thread (program instance); shared memory is shared per thread block. 
    out_vals = x_vals + y_vals

    # Step 6 : Write back the result from the register to the HBM for the current thread/program instance. 
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vector_add_kernel on the provided tensors."""
    n = x.numel()
    BLOCK_SIZE = 1024
    # We can think of grid as number of independent program instances to spawn. 
    # Each independent program instance will run the same code "vector_add_kernel", 
    # BUT, the data in each instance will vary. This is the philosophy of SIMT. 
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)