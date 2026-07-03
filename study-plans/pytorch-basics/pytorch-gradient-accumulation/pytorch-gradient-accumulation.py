import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns:
        (updated_weights_list, last_avg_gradient_list)
    """

    # Initialize learnable weights
    w = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)

    accumulated_grad = torch.zeros_like(w)
    last_avg_grad = None

    for i, (x, y) in enumerate(micro_batches):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # Forward pass
        pred = torch.dot(x, w)
        # MSE loss
        loss = (pred - y) ** 2
        # Compute gradients
        loss.backward()
        # Accumulate gradients
        accumulated_grad += w.grad
        # Clear gradients for next micro-batch
        w.grad.zero_()
        # Update after accum_steps micro-batches
        if (i + 1) % accum_steps == 0:
            avg_grad = accumulated_grad / accum_steps
            last_avg_grad = avg_grad.clone()
            with torch.no_grad():
                w -= lr * avg_grad
            accumulated_grad.zero_()
            
    return w.tolist(), last_avg_grad.tolist()
