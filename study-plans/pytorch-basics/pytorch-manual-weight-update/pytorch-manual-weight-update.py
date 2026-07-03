import torch
import torch.nn as nn

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

    return loss.item()