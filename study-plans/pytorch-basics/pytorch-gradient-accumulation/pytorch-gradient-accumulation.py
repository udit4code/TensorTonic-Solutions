import torch


# The key intuition is that gradient accumulation delays the optimizer step.
# We still compute a gradient for every micro-batch, but we treat several micro-batches as if they formed one larger batch by summing (and then averaging) their gradients before performing a single weight update. This lets us simulate a larger batch size without needing enough memory to process all those samples at once.

# Gradient accumulation is extremely common in real-world deep learning. It's not a niche trick—it exists because GPU memory is limited. Whenever our desired batch size doesn't fit into GPU memory, gradient accumulation is often the first solution.


    
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
            # We use torch.no_grad() because the optimizer update 
            # is not part of the neural network computation that we want to differentiate through.
            # It tells PyTorch : "Under this block, just update the numbers and do not build a computation graph"
            with torch.no_grad():
                w -= lr * avg_grad
            # Doubt : When we do optimizer.step(), why don't we put it inside a no_grad block ? 
            # Because, PyTorch's optimizers already perform parameter updates inside a no-gradient context internally. 
            accumulated_grad.zero_()
            
    return w.detach().tolist(), last_avg_grad.tolist()
