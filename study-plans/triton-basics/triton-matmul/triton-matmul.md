# Tiled Matrix Multiplication

Every loaded element of $A$ in a naive matmul is read $N$ times across the output rows and every element of $B$ is read $M$ times across the output columns. The kernel below is the canonical Triton answer to that arithmetic: a **2D tile** owned by one program, a $K$-loop that accumulates into a register tile, and a single `tl.dot` call per inner iteration that lowers to a tensor-core matmul. The reuse factor is what turns matrix multiplication from a memory-bound operation into a compute-bound one, and the tile is the unit that materializes that reuse.

---

## The Operation

Given $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, the kernel writes $C \in \mathbb{R}^{M \times N}$:

$$
C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]
$$

All three tensors are row-major fp32, the launcher allocates $C$, and the kernel writes into it in place. The total work is $2 M N K$ FLOPs against $4 (M K + K N + M N)$ bytes of HBM traffic at the best case (each element of $A$ and $B$ loaded once and each element of $C$ stored once).

---

## Program Decomposition

Using `BM`, `BN`, and `BK` as compact formula aliases for `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`, the launch grid is two-dimensional, with $\lceil M / \mathrm{BM} \rceil$ programs on axis $0$ and $\lceil N / \mathrm{BN} \rceil$ on axis $1$. Each **program** is identified by the pair $(\mathrm{pm}, \mathrm{pn})$ read from `tl.program_id(0)` and `tl.program_id(1)`, and owns exactly one $\mathrm{BM} \times \mathrm{BN}$ output tile of $C$. The set of owned output tiles partitions the matrix: no two programs write the same element of $C$, and there is no cross-program reduction along $K$ for this version. Output independence makes the kernel embarrassingly parallel at the tile level, even though each tile contains a $K$-long inner reduction the single program must perform sequentially.

The 2D grid mirrors the 2D structure of the output. The parallel pattern is **per-tile reduction**: each program performs a private dot product over $K$ for its tile and stores the result; the only synchronization in the entire kernel is the implicit one between programs in the launcher, which simply waits for all of them to finish before returning control to the host. The compiler will dispatch the programs onto SMs in whatever schedule the runtime sees fit, so the kernel must be correct for any ordering of $(\mathrm{pm}, \mathrm{pn})$ pairs.

---

## Tile Shape and Masking

The tile dimensions are declared `tl.constexpr`, fixed at compile time so the compiler can size registers, unroll the $K$-loop body, and select tensor-core MMA instructions of the right shape. Common starting values are $\mathrm{BM} = \mathrm{BN} = 64$ and $\mathrm{BK} = 32$: powers of two so the compiler can vectorize cleanly, large enough that each `tl.dot` keeps the tensor cores busy, small enough that the per-program register and shared-memory footprint stays inside what the SM exposes.

Three runtime dimensions can overshoot their tile-aligned bounds, so the kernel masks all three. Inside the $K$-loop, `k_mask = (k + offs_k) < K` disables out-of-range columns of the loaded $A$ slab and out-of-range rows of the loaded $B$ slab, with `other = 0.0` so masked lanes contribute additive identities to the accumulator. The final store guards against the $M$ and $N$ tails with `(offs_m[:, None] < M) & (offs_n[None, :] < N)`. Without the $K$ mask the kernel reads past the end of $A$ or $B$ on non-aligned $K$; without the $M$ or $N$ store mask it writes past the end of $C$. Masks here are a correctness obligation, not a performance lever.

---

## Memory Hierarchy and Reuse

Operand tiles begin in HBM. The kernel issues one `tl.load` for the $(\mathrm{BM}, \mathrm{BK})$ slab of $A$ and one for the $(\mathrm{BK}, \mathrm{BN})$ slab of $B$ each $K$-loop iteration; the compiler stages those slabs into on-chip **SRAM** (the on-chip scratchpad the compiler manages for `tl.dot`) and feeds them into the tensor-core MMA instruction. The accumulator `acc` lives in **registers** for the entire $K$-loop and is materialized to HBM only by the final masked store.

The reuse factor is the whole point. Inside one program's $K$-loop iteration, the loaded $(\mathrm{BM}, \mathrm{BK})$ slab of $A$ contributes to every one of the `BLOCK_N` output columns: each fp32 of $A$ is used `BLOCK_N` times. Symmetrically, the loaded $(\mathrm{BK}, \mathrm{BN})$ slab of $B$ contributes to every one of the `BLOCK_M` output rows: each fp32 of $B$ is used `BLOCK_M` times. A naive triple-loop matmul that touches HBM for every multiply would read $A$ and $B$ once per output element, giving zero reuse; tiling transforms the same operation into a memory pattern where every loaded byte performs many FLOPs before being discarded.

L2 reuse exists across programs. Adjacent programs along `pid_n` load the same slab of $A$ in their respective $K$-loops; if their lifetimes overlap on the same SM cluster, the L2 will serve the second program's $A$ slab without going back to HBM. The naive row-major program-ID schedule exploits this poorly; the standard fix is a **grouped program-ID remap** that interleaves `pid_m` and `pid_n` to keep adjacent programs working on overlapping operands. This is a separate optimization and not what this kernel does.

A second compiler-driven memory mechanism that this kernel benefits from implicitly is **software pipelining** of the $K$-loop, controlled by `num_stages`. With `num_stages = 2`, the compiler issues the load for the next iteration's $A$ and $B$ slabs while the current iteration's `tl.dot` is in flight, double-buffering the SRAM staging area so the tensor cores never stall waiting on HBM. The cost is a doubling of SRAM footprint; the benefit is that the latency of operand loads is hidden behind the latency of MMA execution whenever the operand bandwidth and MMA throughput are reasonably balanced.

---

## Memory-Bound vs Compute-Bound

For one $K$-loop iteration, the program loads $\mathrm{BM} \cdot \mathrm{BK} + \mathrm{BK} \cdot \mathrm{BN}$ fp32 values and performs $2 \cdot \mathrm{BM} \cdot \mathrm{BN} \cdot \mathrm{BK}$ FLOPs. The per-iteration arithmetic intensity is therefore

$$
I = \frac{2 \cdot \mathrm{BM} \cdot \mathrm{BN} \cdot \mathrm{BK}}{4 (\mathrm{BM} \cdot \mathrm{BK} + \mathrm{BK} \cdot \mathrm{BN})} = \frac{\mathrm{BM} \cdot \mathrm{BN}}{2 (\mathrm{BM} + \mathrm{BN})} \text{ FLOPs/byte}
$$

For $\mathrm{BM} = \mathrm{BN} = 64$ this evaluates to $16$ FLOPs/byte, well past the typical roofline crossover (around $10$ FLOPs/byte on modern accelerators with $\sim 1$ TB/s HBM and tens of TFLOPs of fp32). The kernel is **compute-bound** with sensible block sizes. The intensity does not depend on `BLOCK_K` in this form because the formula assumes one $K$ iteration; across the full $K$-loop, larger `BLOCK_K` amortizes more arithmetic over each operand load, raising the effective reuse and pushing further past the roofline.

Compare with GEMV, which has no $N$-axis reuse on $x$ and only `BLOCK_M`-fold reuse on each row chunk: intensity is below one FLOP/byte and the kernel sits firmly on the memory side of the roofline. Compare with vector add at $\approx 0.08$ FLOPs/byte, two orders of magnitude lower. Matmul is the operation where the tile model finally produces enough arithmetic per loaded byte to exit the memory-bound regime.

---

## Compiler-Handled vs Author-Handled

`tl.dot` is the most consequential single instruction in Triton, and almost everything about its execution is handled by the compiler. The author writes `acc += tl.dot(a, b)` on a $(\mathrm{BM}, \mathrm{BK})$ and $(\mathrm{BK}, \mathrm{BN})$ pair; the compiler lowers it to a sequence of **tensor-core MMA** instructions of the right shape for the target architecture, allocates the SRAM staging buffers for the two operands, inserts the synchronization needed to coordinate the MMA-issuing warps, and swizzles the operand layout so the bank-conflict pattern that would otherwise serialize SRAM access is avoided. None of that is named in the kernel source.

The author chooses the decomposition: grid shape, tile dimensions, the accumulator dtype (`tl.float32` even when inputs are fp16 or bf16), the mask placement, and the $K$-loop step (`BLOCK_K`). The author also chooses whether to parallelize across the $K$ dimension (a split-K variant that introduces `tl.atomic_add` for the cross-program combine) or keep $K$ private to each program. The compiler picks `num_warps` and `num_stages` at sensible defaults when they are not specified; the author overrides them through autotune when the defaults are wrong for a specific shape.

---

## Naive vs Optimized

The kernel above is already the standard tiled form, and against a per-element implementation it improves HBM traffic by a factor that scales with the smaller of `BLOCK_M` and `BLOCK_N`. A per-element matmul would issue $M N K$ pairs of fp32 loads from HBM ($2 M N K$ loads total) and $M N$ stores; the tiled version issues $\lceil M / \mathrm{BM} \rceil \cdot \lceil N / \mathrm{BN} \rceil \cdot \lceil K / \mathrm{BK} \rceil$ tile-pair loads, each carrying $\mathrm{BM} \cdot \mathrm{BK} + \mathrm{BK} \cdot \mathrm{BN}$ elements, with the same $M N$ stores at the end. The reduction in HBM traffic is roughly a factor of $\min(\mathrm{BM}, \mathrm{BN})$, in line with the reuse factors quoted earlier.

Layered on top, three further optimizations are common in production matmul kernels: **autotune** over the tile shape and pipeline depth (the next problem in this section), **grouped program-ID remap** to exploit L2 reuse between adjacent programs (often called L2-friendly schedule or super-grouping), and `tl.make_block_ptr` with `tl.advance` to express the operand stride math symbolically once and let the compiler emit cleaner pointer arithmetic for the $K$-loop. Each adds a few percent to a couple of times speedup depending on shape; none change the asymptotic roofline placement, which is already compute-bound.

---

## Worked Example

Take $M = N = 4$, $K = 8$, $\mathrm{BM} = \mathrm{BN} = \mathrm{BK} = 2$. The launch grid is $2 \times 2 = 4$ programs, each owning one of the four $2 \times 2$ output tiles. Consider the program at $(\mathrm{pm}, \mathrm{pn}) = (0, 0)$, which owns $C[0{:}2, 0{:}2]$.

The $K$-loop runs $K / \mathrm{BK} = 4$ iterations, with $k = 0, 2, 4, 6$. Each iteration loads a $(2, 2)$ slab of $A$ at rows $[0, 1]$ and columns $[k, k+1]$, a $(2, 2)$ slab of $B$ at rows $[k, k+1]$ and columns $[0, 1]$, calls `tl.dot` producing a $(2, 2)$ partial product, and accumulates into `acc`. After four iterations, `acc` holds $\sum_{k=0}^{7} A[i, k] \cdot B[k, j]$ for $i, j \in \{0, 1\}$, which is exactly the desired output tile.

Counting HBM loads: each iteration reads $4 + 4 = 8$ fp32 values, so the program reads $32$ fp32s of operands across the four iterations and writes $4$ fp32s of output. A per-element implementation of the same tile would read $A[i, k]$ and $B[k, j]$ for each $(i, j, k)$ triple in the tile, $2 \cdot 4 \cdot 8 = 64$ fp32 loads, double the cost. The reuse factor of $\min(\mathrm{BM}, \mathrm{BN}) = 2$ matches the $64 / 32 = 2$ ratio observed here. Scaling the same tile up to $\mathrm{BM} = \mathrm{BN} = 64$ widens the gap: the per-element variant of the same output tile would read $2 \cdot 64 \cdot 64 \cdot \mathrm{BK}$ fp32s while the tiled version reads only $(64 + 64) \cdot \mathrm{BK}$, a $32\times$ traffic reduction.

---

## Why `tl.store` Is Correct Instead of `tl.atomic_add`

Yes: in this kernel, every program eventually writes its completed output tile to global memory. Conceptually, that is an HBM write, although the GPU cache hierarchy may mediate the physical transaction. The important detail is **when** the write happens: the program does not store after every `BK`-wide multiplication. It keeps the output tile in the register accumulator for the entire `K` loop and stores it only after every `K` slice has contributed.

For the program that owns output rows `m0:m1` and output columns `n0:n1`, the computation is conceptually:

```python
acc = zeros(BM, BN)

for k0 in range(0, K, BK):
    a_tile = A[m0:m1, k0:k0 + BK]
    b_tile = B[k0:k0 + BK, n0:n1]
    acc += a_tile @ b_tile

C[m0:m1, n0:n1] = acc
```

An output cell does not need contributions from different output rows or output columns. The cell `C[i, j]` needs the dot product of row `i` from `A` and column `j` from `B`; its only reduction dimension is `K`. Every pass through the loop adds the next `BK`-wide portion of that dot product to the same register cell in `acc`. By the time the loop ends, that accumulator cell contains the complete result for `C[i, j]`.

The launch grid is partitioned only across output rows and output columns. Therefore, exactly one Triton program owns each valid output cell:

- Different `pid_m` values own disjoint groups of output rows.
- Different `pid_n` values own disjoint groups of output columns.
- There is no program-ID axis partitioning `K`.

Because there is exactly one writer per output cell, a normal `tl.store` is both correct and faster than an atomic operation. No other program can race with that store.

An atomic addition becomes necessary in a **split-K** kernel. In that design, multiple programs compute different `K` ranges for the same output tile. Each program then owns only a partial sum, so several programs may update the same `C[i, j]`. Triton's relevant primitive is `tl.atomic_add`, not `tl.add_at`. A split-K implementation must also initialize the output buffer appropriately and accept the synchronization cost and potentially different floating-point accumulation order. Another option is to write partial sums to a separate buffer and launch a second reduction kernel.

The distinction is therefore:

- **This kernel:** one program owns an output tile and reduces over all of `K` internally, then calls `tl.store` once.
- **Split-K kernel:** several programs share an output tile and each reduces over part of `K`, requiring `tl.atomic_add` or a separate reduction pass.

---

## Pen-and-Paper Example: 4 × 4 Matrices with 2 × 2 Tiles

Take the following matrices. The dividing lines show their 2 × 2 blocks. All indices start at zero, and `@` means matrix multiplication.

```text
A =                         B =
[  1   2 |  3   4 ]         [ 1  2 | 0  1 ]
[  5   6 |  7   8 ]         [ 0  1 | 2  0 ]
[ -------+------- ]         [ -----+----- ]
[  9  10 | 11  12 ]         [ 2  0 | 1  2 ]
[ 13  14 | 15  16 ]         [ 1  2 | 0  1 ]
```

We choose `BM = BN = BK = 2`: each program owns a 2 × 2 output tile, and each loop iteration consumes two positions along `K`. Specifying the output tile size alone does not specify `BK`; choosing it to be 2 here makes the arithmetic easy to follow. This is a conceptual walkthrough of the kernel's algorithm, not a runnable tensor-core tile configuration.

### Four Programs Own Four Output Tiles

```text
C =
[ C[0,0]  C[0,1] | C[0,2]  C[0,3] ]
[ C[1,0]  C[1,1] | C[1,2]  C[1,3] ]
[ ---------------+--------------- ]
[ C[2,0]  C[2,1] | C[2,2]  C[2,3] ]
[ C[3,0]  C[3,1] | C[3,2]  C[3,3] ]

Owners:
[ program (0,0) | program (0,1) ]
[ --------------+------------- ]
[ program (1,0) | program (1,1) ]
```

The launch grid is `(2, 2)`. Each of these four programs runs its own two-iteration `K` loop. They may execute concurrently or in any order; none needs another program's result.

### Follow Program (0,0) from Start to Store

This program owns `C[0:2, 0:2]`. Its output rows and columns stay fixed throughout the loop. It begins with a private accumulator:

```text
acc = [ 0  0 ]
      [ 0  0 ]
```

**First iteration: `k_offset = 0`.** Load the first two entries along each relevant row of `A` and each relevant column of `B`:

```text
a = A[0:2, 0:2]        b = B[0:2, 0:2]
  = [ 1  2 ]            = [ 1  2 ]
    [ 5  6 ]              [ 0  1 ]

partial = a @ b
        = [ 1*1 + 2*0    1*2 + 2*1 ]
          [ 5*1 + 6*0    5*2 + 6*1 ]
        = [ 1   4 ]
          [ 5  16 ]

acc = acc + partial = [ 1   4 ]
                      [ 5  16 ]
```

These four values stay in the accumulator. There is no `tl.store` yet because the contributions from `k = 2` and `k = 3` are still missing.

**Second iteration: `k_offset = 2`.** Move the `A` operand tile right by two columns and the `B` operand tile down by two rows. Both partial products contribute to the same output tile:

```text
a = A[0:2, 2:4]        b = B[2:4, 0:2]
  = [ 3  4 ]            = [ 2  0 ]
    [ 7  8 ]              [ 1  2 ]

partial = a @ b
        = [ 3*2 + 4*1    3*0 + 4*2 ]
          [ 7*2 + 8*1    7*0 + 8*2 ]
        = [ 10   8 ]
          [ 22  16 ]

acc = [ 1   4 ] + [ 10   8 ] = [ 11  12 ]
      [ 5  16 ]   [ 22  16 ]   [ 27  32 ]
```

The loop is now finished: all four positions along `K` have contributed. The program executes its final masked `tl.store`, writing these four completed values into `C[0:2, 0:2]`. Every lane is valid because all dimensions divide evenly by 2.

Check one output cell directly using a full row and a full column:

```text
C[0,0] = dot(A[0,:], B[:,0])
       = 1*1 + 2*0 + 3*2 + 4*1
       = (1*1 + 2*0) + (3*2 + 4*1)
       =      1       +     10
       = 11
         iteration 0   iteration 1
```

The row and column are consumed in two pieces, but the same program accumulates both pieces before writing the cell.

### The Other Three Programs Do the Same Work

Each line below shows the partial product from `k_offset = 0`, the partial product from `k_offset = 2`, and the completed tile stored by its owning program:

```text
Program (0,1): C[0:2, 2:4]
[  4  1 ] + [ 3  10 ] = [  7  11 ]
[ 12  5 ]   [ 7  22 ]   [ 19  27 ]

Program (1,0): C[2:4, 0:2]
[  9  28 ] + [ 34  24 ] = [ 43  52 ]
[ 13  40 ]   [ 46  32 ]   [ 59  72 ]

Program (1,1): C[2:4, 2:4]
[ 20   9 ] + [ 11  34 ] = [ 31  43 ]
[ 28  13 ]   [ 15  46 ]   [ 43  59 ]
```

After all four programs have stored their tiles, the output is:

```text
C = A @ B =
[ 11  12 |  7  11 ]
[ 27  32 | 19  27 ]
[ -------+------- ]
[ 43  52 | 31  43 ]
[ 59  72 | 43  59 ]
```

There are eight tile multiplications in total: four programs times two `K` iterations. There are four final tile-store calls: one per program, each writing four output cells. The eight partial products are accumulated privately; they are not eight separate writes to the output. This is why `acc += tl.dot(a, b)` belongs inside the loop and `tl.store(c_ptrs, acc, mask=c_mask)` belongs after it.

---

## Pitfalls

- **Accumulator dtype too narrow.** Declaring `acc` as `tl.float16` silently loses precision once $K$ grows past a few dozen, because the running sum's exponent outpaces the fp16 mantissa's $11$-bit resolution. The accumulator must be `tl.float32` even when the inputs and outputs are lower precision, and the cast to the storage dtype happens only on the final store.

- **Missing the $K$ mask.** When $K$ is not a multiple of `BLOCK_K`, the final $K$-loop iteration overshoots the operand buffers. Without `k_mask = (k + offs_k) < K` on both the $A$ load and the $B$ load, the kernel pulls garbage from past the end of either operand and adds it to the accumulator, breaking every non-aligned shape.

- **Wrong stride order on pointer construction.** Building $A$ pointers with `offs_m` on `stride_ak` and `offs_k` on `stride_am` silently produces $A^{\top} B$ instead of $A B$. The kernel compiles, runs, and may even pass square tests with random inputs that happen to be symmetric. The rule is to pull `A.stride(0), A.stride(1)` from PyTorch and pair them with the row and column index expressions in that order.

- **Block sizes not declared `tl.constexpr`.** A runtime block size forces the compiler to emit a generic loop rather than the unrolled tensor-core MMA sequence. The kernel still produces correct results but loses essentially all the benefit of the tile model, often running an order of magnitude slower.

---
