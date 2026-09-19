import torch
import triton
import triton.language as tl

# LOCATION: GPU KERNEL (Executes in Parallel across Streaming Multiprocessors)
@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Step 1 : Hardware assertion on tile capacity
    # [LOCATION: GPU Registers / SRAM]
    # FIRST PRINCIPLE: Triton maps vectors to fixed-size hardware registers.
    # The number of actual matrix columns (`n_cols`) cannot exceed `BLOCK_SIZE`, 
    # because the kernel loads an entire row into a single SRAM register vector.
    assert n_cols <= BLOCK_SIZE, f"n_cols {n_cols} is strictly greater than BLOCK_SIZE {BLOCK_SIZE}"

    # Step 2 : Program Instance ID assignment
    # [LOCATION: GPU Control Unit / Registers]
    # FIRST PRINCIPLE: Grid Parallelism. The GPU launches `M` parallel instances of this program.
    # `tl.program_id(0)` queries the runtime hardware for the unique index (0, 1, ..., M-1)
    # of the current parallel program. Each program instance processes exactly one row.
    row_idx = tl.program_id(0)

    # Step 3 : SIMD relative column offset vector generation
    # [LOCATION: GPU SRAM Registers]
    # FIRST PRINCIPLE: Vectorized (SIMD) Processing. Instead of looping column-by-column,
    # we instantiate a contiguous sequence vector: [0, 1, 2, ..., BLOCK_SIZE - 1] in SRAM registers.
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Step 4 : Compute base memory pointer for the assigned input row
    # [LOCATION: GPU ALU Computation -> Stored in Registers]
    # FIRST PRINCIPLE: Strided Memory Flattening. 2D matrices are stored in continuous 1D GPU HBM.
    # To reach row `row_idx`, we multiply `row_idx` by `x_row_stride` (the memory gap between adjacent rows).
    x_start_ptr = x_ptr + row_idx * x_row_stride

    # Step 5 : Compute exact global memory pointers for all elements in this row
    # [LOCATION: GPU SRAM Registers]
    # FIRST PRINCIPLE: Vector Pointer Arithmetic. Adding the scalar base pointer `x_start_ptr` 
    # to the offset vector `col_offsets` generates an array of memory addresses pointing directly to HBM locations.
    x_ptrs = x_start_ptr + col_offsets

    # Step 6 : Construct memory boundary predicate mask
    # [LOCATION: GPU Predicate Registers]
    # FIRST PRINCIPLE: Power-of-2 Memory Padding. GPUs operate most efficiently when block sizes 
    # are powers of two (e.g., 2, 4, 8, ... 1024). If `N=3`, `BLOCK_SIZE=4`.
    # `row_mask` becomes `[True, True, True, False]`, preventing invalid reads past column `n_cols`.
    row_mask = col_offsets < n_cols

    # Step 7 : Fetch row vector from slow HBM into ultra-fast SRAM / Registers
    # [LOCATION: Data Transfer from GPU HBM (Global DRAM) -> GPU SRAM Registers]
    # FIRST PRINCIPLE: Neutral Identity Elements for Max Reductions.
    # Unallocated/padded columns where `row_mask` is `False` are populated with `-inf`.
    # Since max(x, -inf) = x, padding with `-inf` guarantees that dummy lanes never skew the maximum.
    x_row = tl.load(x_ptrs, mask=row_mask, other=float("-inf"))

    # Step 8 : Parallel tree-reduction to find the row maximum
    # [LOCATION: GPU SRAM / Local Execution Registers]
    # FIRST PRINCIPLE: Floating-Point Overflow Protection.
    # LogSumExp requires calculating exp(x). For x = 1000, e^1000 causes immediate floating-point overflow (inf).
    # Mathematically: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x)))).
    # Subtracting the maximum forces all exponents to be <= e^0 = 1.0, making the computation 100% numerically stable.
    row_max = tl.max(x_row, axis=0)

    # Step 9 : Compute shift and exponentiate valid row entries
    # [LOCATION: GPU Floating-Point Units (ALUs) -> Registers]
    # FIRST PRINCIPLE: Vectorized Exponential Evaluation.
    # Subtracts `row_max` from each element and applies the natural exponential function e^(x_i - max).
    exp_row = tl.exp(x_row - row_max)
    
    # Step 10 : Clean padded lanes before summation
    # [LOCATION: GPU SRAM Registers]
    # FIRST PRINCIPLE: Indeterminate Form Prevention & Additive Identity.
    # For padded lanes where x_row was `-inf`, `-inf - row_max` remains `-inf`, yielding exp(-inf) = 0.0.
    # However, if ALL values in a row are `-inf`, `-inf - (-inf)` yields NaN!
    # `tl.where` explicitly replaces padded lanes with 0.0 (the additive identity element: x + 0 = x).
    exp_row = tl.where(row_mask, exp_row, 0.0)

    # Step 11 : Parallel tree-reduction sum of exponentiated values
    # [LOCATION: GPU SRAM / Local Execution Registers]
    # FIRST PRINCIPLE: Intra-Block Parallel Tree Reduction.
    # Sums all exponentials across the row vector to yield a single scalar `row_exp_sum`.
    row_exp_sum = tl.sum(exp_row, axis=0)

    # Step 12 : Compute final LogSumExp scalar result
    # [LOCATION: GPU SRAM Registers]
    # FIRST PRINCIPLE: Mathematical Reconstruction.
    # Evaluates log(sum(exp(x - max))) and adds back `row_max` to restore the true mathematical value.
    row_log_exp_sum = row_max + tl.log(row_exp_sum)

    # Step 13 : Write final result from local SRAM back to GPU Global Memory (HBM)
    # [LOCATION: Data Transfer from GPU SRAM Registers -> GPU HBM (Global DRAM)]
    # FIRST PRINCIPLE: 1D Contiguous Memory Layout.
    # Since `out` is a 1D tensor of shape `(M,)`, the destination pointer for row `row_idx` 
    # is directly `out_ptr + row_idx`. `tl.store` writes the scalar result back to global memory.
    out_start_ptr = out_ptr + row_idx 
    tl.store(out_start_ptr, row_log_exp_sum)



# LOCATION: HOST DEVICE CPU (Python Execution & GPU Launch Management)
def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    # Step 14 : Extract tensor shape metadata
    # [LOCATION: HOST CPU]
    # FIRST PRINCIPLE: Tensor Geometry. `M` is the total number of rows, `N` is the column count.
    M, N = x.shape

    # Step 15 : Hardware alignment rounding
    # [LOCATION: HOST CPU]
    # FIRST PRINCIPLE: Power-of-2 Memory Alignment. Computes the smallest power of 2 
    # that is greater than or equal to `N` (e.g., if N=3, BLOCK_SIZE=4).
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Step 16 : Define grid dimension layout
    # [LOCATION: HOST CPU]
    # FIRST PRINCIPLE: Task Decomposition. `grid = (M,)` instructs the GPU driver to dispatch 
    # `M` independent parallel program instances, assigning 1 thread block per matrix row.
    grid = (M,)

    # Step 17 : Dispatch kernel execution to GPU stream
    # [LOCATION: HOST CPU -> PCIe Driver -> GPU Hardware Scheduler]
    # FIRST PRINCIPLE: Asynchronous Kernel Dispatch.
    # Passes memory pointers (`x`, `out`), row stride (`x.stride(0)`), and scalar values to the GPU device.
    # CPU queues the kernel on the CUDA command stream and continues execution immediately.
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )