import torch
import torch.nn as nn


# Each learnable parameter (weights and biases) stores its own gradient in a .grad field. Calling loss.backward() adds the newly computed gradient to whatever is already in that field. Clearing gradients with model.zero_grad() or optimizer.zero_grad() empties those buffers so the next training iteration computes gradients only for the current mini-batch. The main exception is intentional gradient accumulation, where you deliberately leave the gradients in place across several micro-batches before performing a single optimizer step.
    
def manual_train_step(model, X, y, criterion, lr):
    """
        Returns: loss value as a Python float
    """
    # Put model in training mode
    model.train()
    # Forward pass
    predictions = model(X)
    # Compute loss
    loss = criterion(predictions, y)
    # Backward pass
    loss.backward()
    # Manually update each parameter inside a no_grad() block. 
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None: 
                param -= lr * param.grad
            else:
                print(f"param {param} has NONE gradient")

    # Clear gradients
    for param in model.parameters():
        if param.grad is not None: 
            param.grad.zero_()
    # We can use model.zero_grad() instead of the above block 
    return loss.item()