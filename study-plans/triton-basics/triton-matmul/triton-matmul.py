import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    indices_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (indices_m[:, None] * stride_am) + (indices_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (indices_k[:, None] * stride_bk) + (indices_n[None, :] * stride_bn)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_offset in range(0, K, BLOCK_K):
        a_mask = (indices_m[:, None] < M) & ((k_offset + indices_k[None, :]) < K)
        b_mask = ((k_offset + indices_k[:, None]) < K) & (indices_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Cast to float16 to ensure Tensor Core compatibility on all GPUs
        a = a.to(tl.float16)
        b = b.to(tl.float16)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (indices_m[:, None] * stride_cm) + (indices_n[None, :] * stride_cn)
    c_mask = (indices_m[:, None] < M) & (indices_n[None, :] < N)

    # Convert accumulator back to fp16 if your output tensor expects it
    tl.store(c_ptrs, acc, mask=c_mask)

def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch matmul_kernel: out = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible matrix dimensions: K1={K}, K2={K2}"
    
    # Ensure inputs are contiguous in memory depending on your strides
    assert A.is_contiguous(), "Matrix A must be contiguous"
    assert B.is_contiguous(), "Matrix B must be contiguous"

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )