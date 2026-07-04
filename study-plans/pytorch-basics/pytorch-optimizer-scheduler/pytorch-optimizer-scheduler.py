import torch
import torch.nn as nn

import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion,optimizer, scheduler, num_epochs):

    losses = []
    lrs = []
    for epoch in range(num_epochs):
        # Put the model into training mode.
        # This enables the training behaviour of layers such as Dropout and BatchNorm.
        model.train()

        running_loss = 0.0
        for X, y in dataloader:
            # Clear gradients from the previous iteration.
            # PyTorch accumulates gradients by default.
            optimizer.zero_grad()

            # Forward pass.
            predictions = model(X)
            # Compute the loss for this mini-batch.
            loss = criterion(predictions, y)
            # Compute gradients with respect to all learnable parameters.
            loss.backward()
            # Update the model parameters using the optimizer's current learning rate.
            optimizer.step()

            running_loss += loss.item()
        # Compute the average loss over all mini-batches in this epoch.
        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        # Record the learning rate that was used during the current epoch.
        lrs.append(optimizer.param_groups[0]["lr"])

        # Advance the learning-rate scheduler.
        # The scheduler updates the optimizer's internal learning rate
        # according to its scheduling policy (StepLR, CosineAnnealingLR,
        # ExponentialLR, etc.).
        #
        # For StepLR, once every 'step_size' epochs:
        #
        #     new_lr = old_lr * gamma
        #
        # The updated learning rate will be used by
        # optimizer.step() in the NEXT epoch.
        scheduler.step()

    return {
        "losses": losses,
        "lrs": lrs
    }