"""Illustrate block addressing and CUDA metadata transfer, not a serving engine.

Run with --device cpu to verify the arithmetic without a CUDA GPU.
The CUDA path uses pinned host metadata, two streams, and an event dependency.
Toy scalar K/V stand in for one layer and one attention head.
"""
import argparse
from contextlib import nullcontext

import torch


def run(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu for the arithmetic demo.")
    device = torch.device(device_name)
    use_cuda = device.type == "cuda"
    block_size = 2

    # In a server, these pools persist across requests. Here: [blocks, tokens, dim].
    k_pool = torch.zeros((201, block_size, 1), device=device)
    v_pool = torch.zeros_like(k_pool)
    # Simulate READY state produced by a previous request, already on the device.
    k_pool[102, :, 0] = torch.tensor([0.1, 0.2], device=device)
    k_pool[105, :, 0] = torch.tensor([0.3, 0.4], device=device)
    v_pool[102, :, 0] = torch.tensor([1.0, 2.0], device=device)
    v_pool[105, :, 0] = torch.tensor([3.0, 4.0], device=device)
    prefix_before = k_pool[[102, 105]].clone()

    # CPU scheduler's plan: only metadata and new token IDs, not prefix tensors.
    fields = {
        "tokens": [99, 88],
        "positions": [4, 5],
        "block_table": [102, 105, 200],
        "write_slots": [400, 401],  # physical block 200 * 2 + offset
    }
    host = {}
    gpu = {}
    for name, values in fields.items():
        host[name] = torch.empty(len(values), dtype=torch.int64, pin_memory=use_cuda)
        host[name].copy_(torch.tensor(values, dtype=torch.int64))
        gpu[name] = torch.empty(len(values), dtype=torch.int64, device=device)

    copy_stream = torch.cuda.Stream() if use_cuda else None
    compute_stream = torch.cuda.Stream() if use_cuda else None
    if use_cuda:
        # Also order side-stream work after setup/allocation on the current stream.
        copy_stream.wait_stream(torch.cuda.current_stream())
        compute_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(copy_stream) if use_cuda else nullcontext():
        for name in fields:
            gpu[name].copy_(host[name], non_blocking=use_cuda)
        if use_cuda:
            metadata_ready = torch.cuda.Event()
            metadata_ready.record(copy_stream)

    with torch.cuda.stream(compute_stream) if use_cuda else nullcontext():
        if use_cuda:
            compute_stream.wait_event(metadata_ready)
        # Synthetic K/V, computed on the device. A real model uses projections
        # of hidden states at every layer, with position handling as appropriate.
        new_k = gpu["tokens"].float().unsqueeze(1) / 100
        new_v = gpu["tokens"].float().unsqueeze(1) / 10
        k_pool.view(-1, 1).index_copy_(0, gpu["write_slots"], new_k)
        v_pool.view(-1, 1).index_copy_(0, gpu["write_slots"], new_v)

        # Read logical positions 0..5 through the request's block table.
        logical_positions = torch.arange(6, device=device)
        physical_blocks = gpu["block_table"][logical_positions // block_size]
        read_slots = physical_blocks * block_size + logical_positions % block_size
        # Teaching-only gather: this COPIES data within the device. Production
        # paged attention loads tiles directly, avoiding this materialization.
        keys = k_pool.view(-1, 1).index_select(0, read_slots)
        values = v_pool.view(-1, 1).index_select(0, read_slots)
        queries = torch.ones((2, 1), device=device)
        scores = queries @ keys.T  # head dimension 1, so sqrt(d) = 1
        mask = logical_positions.unsqueeze(0) <= gpu["positions"].unsqueeze(1)
        output = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ values
        if use_cuda:
            finished = torch.cuda.Event()
            finished.record(compute_stream)

    # Keep host buffers, device buffers, and pools alive and unmodified until done.
    if use_cuda:
        finished.synchronize()  # Deliberate blocking boundary for this demo.
    expected_slots = torch.tensor([204, 205, 210, 211, 400, 401])
    torch.testing.assert_close(read_slots.cpu(), expected_slots)
    torch.testing.assert_close(k_pool[[102, 105]], prefix_before)
    reference_tokens = torch.tensor([10, 20, 30, 40, 99, 88], dtype=torch.float32)
    reference_scores = (reference_tokens / 100).repeat(2, 1)
    reference_scores[0, 5] = float("-inf")  # Position 4 cannot see position 5.
    reference = reference_scores.softmax(-1) @ (reference_tokens / 10).unsqueeze(1)
    torch.testing.assert_close(output.cpu(), reference)
    print(f"Device: {device}; metadata payload: 72 bytes (excluding launch overhead)")
    print("Physical read slots:", read_slots.cpu().tolist())
    print("Attention outputs:", output.cpu().flatten().tolist())
    print("PASS: reused prefix unchanged; paged lookup matches contiguous reference.")
    if not use_cuda:
        print("CPU mode checks arithmetic only; it does not exercise CUDA transfers or streams.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run(parser.parse_args().device)
