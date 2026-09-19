import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    assert n_cols <= BLOCK_SIZE, f"n_cols {n_cols} is strictly greater than BLOCK_SIZE {BLOCK_SIZE}"
    # Step 1 : Identify which matrix row does this current GPU program instance will compute 
    row_idx = tl.program_id(0)

    # Step 2 : Generate relative column indices for vectorized processing
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Step 3 : Get the base memory address (pointer) for the start of the assigned row. 
    x_start_ptr = x_ptr + row_idx * x_row_stride

    # Step 4 : Compute exact global addresses for every column in this row.
    x_ptrs = x_start_ptr + col_offsets

    # Step 5 : Compute the row_mask to prevent out-of-bounds access
    row_mask = col_offsets < n_cols

    # Step 6 : Load the row data from HBM to SRAM. 
    x_row = tl.load(x_ptrs, mask=row_mask, other=float("-inf"))

    # Step 6: Compute max across row for numerical stability
    row_max = tl.max(x_row, axis=0)

    # Step 7: Exponentiate and sum valid elements
    # Subtracting row_max guarantees exponents fit inside float32 range without overflowing
    exp_row = tl.exp(x_row - row_max)
    
    # Mask out padded lanes before sum so exp(-inf - row_max) -> 0.0 doesn't cause floating issues
    exp_row = tl.where(row_mask, exp_row, 0.0)
    row_exp_sum = tl.sum(exp_row, axis=0)

    # Step 8: Complete the LogSumExp formula: max + log(sum(exp(x - max)))
    row_log_exp_sum = row_max + tl.log(row_exp_sum)

    # Step 9 : Store row result directly to its assigned index in output array
    out_start_ptr = out_ptr + row_idx 
    tl.store(out_start_ptr, row_log_exp_sum)



def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )
