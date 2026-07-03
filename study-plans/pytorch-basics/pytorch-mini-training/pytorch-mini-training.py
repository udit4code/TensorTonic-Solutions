import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
        Performs one training epoch.
        Args:
            model: PyTorch model
            dataloader: DataLoader yielding (inputs, targets)
            criterion: Loss function
            optimizer: Optimizer
        Returns:
            float: Average loss over all batches.
    """
    model.train()

    total_loss = 0.0

    for inputs, targets in dataloader:
        # Step 1: Clear gradients stored on the model parameters.
        # Gradients accumulate by default in PyTorch, so they must be
        # reset before computing gradients for the current batch.
        optimizer.zero_grad()
        # Step 2: Run the forward pass to compute predictions.
        prediction_outputs = model(inputs)
        # Step 3: Compute the loss between predictions and ground-truth labels.
        loss = criterion(prediction_outputs, targets)
        # Step 4: Backpropagate to compute gradients of the loss
        # with respect to all learnable parameters.
        loss.backward()
        # Step 5: Update the model parameters using the computed gradients.
        optimizer.step()
        # Add this batch's loss to the running total.
        total_loss += loss.item()
    # Return the average batch loss for this epoch.
    return total_loss / len(dataloader)
