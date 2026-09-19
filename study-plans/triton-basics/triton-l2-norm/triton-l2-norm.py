import torch
import triton
import triton.language as tl


# KEY CONCEPT: 1D Reduction across GPU Thread Blocks : 
# Problem: We want to compute sqrt(sum(x_i^2)) for a large 1D tensor of size `n`.
# Challenge: A single thread block cannot process millions of elements at once due to
# hardware limits on SRAM/register allocation and vector width.
# Solution (Grid-Stride / Tiled Parallel Reduction):
#   1. Slice the 1D tensor into contiguous chunks of size `BLOCK_SIZE`.
#   2. Launch multiple parallel program instances (Grid size = number of chunks).
#   3. Each program reads its chunk from HBM -> computes local sum of squares in SRAM registers.
#   4. Every program atomically updates a SINGLE 1-element buffer in global GPU memory (HBM).
#   5. The host script launches a follow-up PyTorch sqrt on the GPU to finish the L2 norm.


@triton.jit
def l2_norm_kernel(x_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Step 1 : Identify tile identity in the Grid
    # FIRST PRINCIPLE: Massive Parallelism. `tl.program_id(0)` returns the 0-indexed ID 
    # of the current thread block in dimension 0 of our 1D grid layout.
    # Program 0 handles chunk [0 : BLOCK_SIZE], Program 1 handles chunk [BLOCK_SIZE : 2*BLOCK_SIZE], etc.
    tile_idx = tl.program_id(0)

    # Step 2 : Compute global base offset for this tile
    # FIRST PRINCIPLE: Pointer Arithmetic in 1D Arrays. Linear address calculation:
    # Starting offset = (tile index) * (elements per tile).
    # Adding this integer offset to base pointer `x_ptr` targets the start of our chunk.
    x_start_ptr = x_ptr + tile_idx * BLOCK_SIZE

    # Step 3 : Generate SIMD relative offset indices
    # FIRST PRINCIPLE: SIMD (Single Instruction, Multiple Data). GPUs operate on 
    # vector lanes. `tl.arange(0, BLOCK_SIZE)` creates a hardware register array:
    # [0, 1, 2, ..., BLOCK_SIZE - 1].
    offsets = tl.arange(0, BLOCK_SIZE)

    # Combine scalar base pointer with vector offsets to get an array of pointers:
    # x_ptrs = [x_start_ptr + 0, x_start_ptr + 1, ..., x_start_ptr + BLOCK_SIZE - 1]
    x_ptrs = x_start_ptr + offsets

    # Step 4 : Predicate masking for out-of-bounds safety
    # FIRST PRINCIPLE: Boundary Protection. If input size `n` is not an exact multiple 
    # of `BLOCK_SIZE`, the last tile will read past valid array memory.
    # `tile_idx * BLOCK_SIZE + offsets < n` creates a boolean vector mask.
    # Elements where index >= n evaluate to False.
    mask = (tile_idx * BLOCK_SIZE + offsets) < n 

    # Step 5 : Load data from Global Memory (HBM) into Local SRAM Registers
    # FIRST PRINCIPLE: Memory Hierarchy & Neutral Identity for Additive Reductions.
    # Loads vector from slow HBM to fast SM SRAM registers.
    # Padded lanes where mask == False are filled with `other = 0.0`.
    # Why 0.0? Because 0.0^2 = 0.0, which acts as the additive identity element (x + 0 = x),
    # ensuring padding contributes zero to the final squared sum without polluting results.
    data = tl.load(x_ptrs, mask=mask, other=0.0)

    # Step 6 : Multiply and Sum Reduction in Fast Local Registers
    # FIRST PRINCIPLE: Fused Register Computation vs. Memory Materialization.
    # `data * data` computes element-wise squares inside GPU execution registers.
    # `tl.sum(..., axis=0)` applies a tree-based hardware reduction across the block 
    # to compress the vector down to a single 32-bit float scalar (`block_squared_sum`).
    # By fusing these, we avoid writing intermediate `data_squared` tensors to memory, 
    # maximizing register reuse and drastically saving memory bandwidth.
    block_squared_sum = tl.sum(data * data, axis=0)


    # Step 7 : Accumulate block scalar into Global HBM Buffer via Atomic Addition
    # FIRST PRINCIPLE: Race Condition Prevention & Memory Interconnect Routing.
    # ALL thread blocks in the grid execute concurrently and share a single output 
    # accumulator at address `sumsq_ptr`. 
    # Standard write (`*sumsq_ptr += block_squared_sum`) causes a classic Data Race.
    # `tl.atomic_add` uses hardware-level memory lock primitives directly on GPU HBM/L2 
    # cache to serialize concurrent updates from parallel blocks.
    # CRITICAL INTERVIEW NOTE: The data transfer moves from SRAM -> GPU Global Memory (HBM).
    # NO host CPU bus transfer (PCIe) occurs during this operation.
    tl.atomic_add(sumsq_ptr, block_squared_sum)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Host launch controller and final computation stage."""
    # Step 1 : Read input tensor size
    n = x.numel()

    # Step 2 : Allocate on-device global accumulator buffer
    # FIRST PRINCIPLE: Zero-Initialization for Atomic Accumulators.
    # Must be initialized to 0.0 on GPU device (`cuda`). Atomic add performs `buf = buf + value`.
    # Uninitialized garbage values in memory would invalidate final sums.
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)


    # Step 3 : Hardware Tile Sizing & Grid Size Calculation
    # FIRST PRINCIPLE: Occupational Optimization.
    # 1024 is standard max block size for modern GPUs (1024 threads per thread block).
    # Grid formula `(n + BLOCK_SIZE - 1) // BLOCK_SIZE` computes ceiling division (ceil(n / BLOCK_SIZE))
    # to guarantee enough blocks are launched to process all elements.
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Step 4 : Kernel Launch Dispatch
    # Compiles and queues `grid` kernel instances on GPU streams via CUDA driver.
    l2_norm_kernel[grid](x, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)

    # Step 12 : Final Sqrt Post-Processing
    # FIRST PRINCIPLE: Asynchronous GPU Execution & Final Reduction Stage.
    # `torch.sqrt` runs on GPU VRAM, taking square root of accumulated sum of squares.
    # `out.copy_(...)` fills destination tensor memory on GPU.
    out.copy_(torch.sqrt(sumsq_buf))