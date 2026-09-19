import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions: A is (M x K), B is (K x N), C is (M x N)
    M, N, K,
    # Strides allow us to navigate memory correctly (how much to jump to get to the next row/col)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters for tile sizes (must be known at compile time)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Step 1 : GRID IDENTIFICATION
    # Each Triton program instance computes one [BLOCK_M x BLOCK_N] tile of the output matrix C.
    # We find out WHICH tile this specific program is computing using program IDs.
    # DOUBT : Why are we calling tl.program_id twice ? 
    # tl.program_id(axis) returns a single integer, not a tuple. 
    # It returns the coordinate of the current GPU program instance along the specified dimension (axis) of our execution grid.
    # The mental model is : when we launch matmul_kernel[(grid_m, grid_n)], Triton spins up grid_m * grid_n independent program instances. Each program instance is assigned to compute exactly one cell of that grid.
    # We have to call tl.program_id(axis) twice, because, we are fetching coordinates, not different IDs.
    # In other words, we are not getting two different program IDs; 
    # we are getting the X and Y coordinates of the same program ID. 
    # Because we launched a 2D grid, every program instance exists on a 2D plane. 
    # To know exactly which chunk of the matrix it needs to process, the program needs to know its row and its column.
    # If we were writing a simple kernel to add two 1D vectors together, we would launch a 1D grid: grid = (num_blocks,). 
    # In that kernel, we would only call tl.program_id(0) because a 1D line only has one axis.
    # Note for CUDA backgrounds: tl.program_id(0) is exactly equivalent to blockIdx.x in CUDA C++, and tl.program_id(1) is blockIdx.y
    # Because matrix multiplication produces a 2D output matrix, it is mathematically much easier to map it to a 2D execution grid, which requires checking both axis 0 and axis 1.
    
    pid_m = tl.program_id(0) # Row index of the tile
    pid_n = tl.program_id(1) # Column index of the tile


    # Step 2 : GENERATE 1D INDICES
    # Create arrays of offsets for the current block. 
    # Example: If pid_m=1 and BLOCK_M=64, indices_m = [64, 65, ..., 127]
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # The K dimension is handled in a loop, so we just need a base vector from 0 to BLOCK_K
    indices_k = tl.arange(0, BLOCK_K)

    # Step 3 : COMPUTE 2D MEMORY POINTERS
    # We broadcast the 1D indices to 2D matrices of pointers.
    # `indices_m[:, None]` creates a column vector, `indices_k[None, :]` creates a row vector.
    # Adding them with their respective strides creates a 2D grid of memory addresses.
    a_ptrs = a_ptr + (indices_m[:, None] * stride_am) + (indices_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (indices_k[:, None] * stride_bk) + (indices_n[None, :] * stride_bn)

    # Step 4 : INITIALIZE ACCUMULATOR
    # Allocate a [BLOCK_M x BLOCK_N] grid in ultra-fast SRAM to hold the running sum.
    # We use float32 for accumulation to prevent precision loss.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Step 5 : INNER K-LOOP (TILE REDUCTION)
    # Sweep along the K dimension of A and B, loading [BLOCK_M x BLOCK_K] from A
    # and [BLOCK_K x BLOCK_N] from B in each iteration.
    for k_offset in range(0, K, BLOCK_K):
        # Create masks to prevent reading out of bounds if M, N, or K are not perfect multiples of block sizes.
        a_mask = (indices_m[:, None] < M) & ((k_offset + indices_k[None, :]) < K)
        b_mask = ((k_offset + indices_k[:, None]) < K) & (indices_n[None, :] < N)

        # Load tiles from High Bandwidth Memory (HBM) into SRAM. 
        # Out-of-bounds elements are padded with 0.0.
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Cast to float16 to utilize hardware Tensor Cores. 
        # Tensor Cores strictly require fp16, bf16, or int8 inputs on most architectures.
        a = a.to(tl.float16)
        b = b.to(tl.float16)

        # Matrix Multiply-Accumulate (MAC). 
        # Mathematically: acc += A_tile @ B_tile
        acc += tl.dot(a, b)

        # Advance the pointer grids to the next block along the K dimension.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Step 6 : WRITE BACK TO HBM
    # Calculate exactly where in matrix C this program should write its finished tile.
    c_ptrs = c_ptr + (indices_m[:, None] * stride_cm) + (indices_n[None, :] * stride_cn)
    
    # Mask to prevent writing out of bounds on the edges of the matrix.
    c_mask = (indices_m[:, None] < M) & (indices_n[None, :] < N)

    # Store the accumulated [BLOCK_M x BLOCK_N] tile back to global memory.
    # If the output tensor expects float16, Triton will implicitly cast the float32 accumulator here.
    tl.store(c_ptrs, acc, mask=c_mask)


# HOST DEVICE CPU (Launch Setup)
def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch matmul_kernel: out = A @ B."""
    
    # Get matrix dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible matrix dimensions: K1={K}, K2={K2}"
    
    # Safety check: Basic Triton pointers logic assumes matrices are contiguous in memory.
    assert A.is_contiguous(), "Matrix A must be contiguous"
    assert B.is_contiguous(), "Matrix B must be contiguous"

    # Define hyper-parameters (Tile sizes). 
    # In production, these are often autotuned using @triton.autotune
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Define the 2D execution grid. 
    # triton.cdiv performs ceiling division (e.g., cdiv(100, 64) = 2).
    # This guarantees we launch enough blocks to cover the entire M x N output matrix.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel on the GPU
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1), # Strides for Matrix A
        B.stride(0), B.stride(1), # Strides for Matrix B
        out.stride(0), out.stride(1), # Strides for Matrix C (out)
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )