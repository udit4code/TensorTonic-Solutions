import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Step 1 : Validate block size hardware capability
    # FIRST PRINCIPLE: GPUs process data in fixed-size blocks (vectors) rather than element-by-element loops.
    # Because Triton loads an entire row into a fixed-length memory array of size `BLOCK_SIZE`, 
    # the total number of columns (`n_cols`) in our actual input matrix cannot exceed this hardware block capacity.
    assert n_cols <= BLOCK_SIZE, f"n_cols {n_cols} is greater than block size {BLOCK_SIZE}"

    # Step 2 : Identify which matrix row this specific GPU program instance will compute
    # FIRST PRINCIPLE: Massive parallelism. Instead of one CPU thread looping through all rows sequentially,
    # the GPU launches thousands of tiny programs in parallel. `tl.program_id(0)` returns the unique ID 
    # (0, 1, 2, ...) of the current parallel program instance. We map 1 program ID directly to 1 matrix row.
    row_idx = tl.program_id(0)

    # Step 3 : Generate relative column indices for vector processing
    # FIRST PRINCIPLE: SIMD (Single Instruction, Multiple Data). Rather than reading columns one by one,
    # we create a contiguous sequence of column indices: [0, 1, 2, ..., BLOCK_SIZE - 1].
    # This allows the GPU to compute memory addresses for an entire vector simultaneously.
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Step 4 : Calculate the base memory address (pointer) for the start of the assigned row
    # FIRST PRINCIPLE: Flattened 2D Memory Storage. A 2D matrix in memory is stored as a single continuous 1D array.
    # To skip to row `row_idx`, we multiply `row_idx` by `x_row_stride` (the number of memory elements to skip to reach the next row).
    x_start_ptr = x_ptr + row_idx * x_row_stride

    # Step 5 : Compute exact global memory addresses (pointers) for every column in this row
    # FIRST PRINCIPLE: Pointer arithmetic. Adding the vector of column offsets [0, 1, 2, ...] to the scalar `x_start_ptr`
    # produces a vector of distinct memory addresses pointing directly to each element in the row.
    x_ptrs = x_start_ptr + col_offsets

    # Step 6 : Create a memory boundary mask to prevent out-of-bounds access
    # FIRST PRINCIPLE: Power-of-2 Memory Padding. GPUs operate most efficiently when block sizes are powers of 2 (e.g., 128, 256, 512).
    # If our matrix has 300 columns, `BLOCK_SIZE` will be 512. The boolean mask will be `True` for columns 0..299 
    # and `False` for padded columns 300..511, ensuring we don't read unallocated memory.
    row_mask = col_offsets < n_cols

    # Step 7 : Fetch row data from slow High Bandwidth Memory (HBM) into ultra-fast SRAM registers
    # FIRST PRINCIPLE: Numerically neutral masking values for Softmax. Softmax relies on exponents: exp(x).
    # Since exp(-infinity) = 0, setting out-of-bounds padded values (`other=float("-inf")`) ensures they contribute
    # zero to both the numerator and denominator, leaving the true mathematical result completely undisturbed.
    x_row = tl.load(x_ptrs, mask=row_mask, other=float("-inf"))

    # Step 8 : Subtract the maximum value in the row to guarantee numerical stability
    # FIRST PRINCIPLE: Floating-point overflow protection. In standard Softmax, exp(x) grows exponentially.
    # For large inputs like x = 1000, e^1000 causes floating-point overflow (inf).
    # Mathematically, exp(x_i) / sum(exp(x)) == exp(x_i - max) / sum(exp(x - max)).
    # Subtracting the row max forces all exponents to be <= exp(0) = 1.0, completely avoiding overflow.
    # In Triton, tl.max(x_row, axis=0) finds the maximum (largest) value in the row vector currently held in the GPU's fast registers.
    # In Triton, axis=0 specifies the dimension along which to perform the reduction. Since x_row is a 1D tensor, axis=0 reduces all the elements across its single dimension down to a single scalar maximum value.
    x_row = x_row - tl.max(x_row, axis=0)

    # Step 9 : Compute the numerator of the Softmax equation (element-wise exponential)
    # FIRST PRINCIPLE: Element-wise vector operations. The GPU applies the natural exponential function e^(x_i - max)
    # to every valid column in the row simultaneously in fast local registers (SRAM).
    numerator = tl.exp(x_row)

    # Step 10 : Compute the denominator of the Softmax equation (sum of exponentials)
    # FIRST PRINCIPLE: Parallel Reduction. All exponential values across the row vector are summed together 
    # to produce a single scalar normalization factor for this row.
    denominator = tl.sum(numerator, axis=0)

    # Step 11 : Normalize values to produce the final probability distribution
    # FIRST PRINCIPLE: Probability Axioms. Dividing each exponential by the sum of all exponentials scales 
    # all outputs into the range [0.0, 1.0], ensuring the entire row sums to exactly 1.0.
    y_row = numerator / denominator

    # Step 12 : Compute output memory pointers for storing results back to HBM
    # FIRST PRINCIPLE: Parallel writes. Calculate the exact global HBM memory addresses for the output matrix `y`
    # using the same row stride and column offset arithmetic applied during the input step.
    y_start_ptr = y_ptr + row_idx * y_row_stride 
    y_ptrs = y_start_ptr + col_offsets

    # Step 13 : Write computed Softmax results back to global GPU memory
    # FIRST PRINCIPLE: Guarded memory writes. Using `mask=row_mask` guarantees that the GPU writes results ONLY 
    # for valid data columns (0..n_cols-1) and discards dummy padded values, preserving tensor integrity.
    tl.store(y_ptrs, y_row, mask=row_mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    # Step 14 : Extract tensor dimensions
    # FIRST PRINCIPLE: Matrix representation. `M` represents the number of independent rows (batch/sequence size),
    # and `N` represents the number of columns per row.
    M, N = x.shape

    # Step 15 : Round up the column count to the nearest power of 2
    # FIRST PRINCIPLE: GPU hardware alignment. Tensor Core architectures execute operations most efficiently 
    # when vector lengths are powers of 2 (e.g., 32, 64, 128, 256, 512, 1024).
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Step 16 : Define the execution grid size
    # FIRST PRINCIPLE: Grid decomposition. A grid specifies how many parallel program instances to launch on the GPU.
    # Setting `grid = (M,)` launches `M` concurrent program instances, granting each row its own dedicated GPU worker.
    grid = (M,)

    # Step 17 : Launch the compiled Triton kernel on the GPU
    # FIRST PRINCIPLE: Host-to-Device kernel dispatch. PyTorch passes memory pointers, strides, shapes, 
    # and compile-time constants (`BLOCK_SIZE`) to Triton's JIT compiler, executing the custom kernel on hardware.
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )