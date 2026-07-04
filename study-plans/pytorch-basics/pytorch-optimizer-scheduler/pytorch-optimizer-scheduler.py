import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
        Returns:
            {
                "losses": [...],
                "lrs": [...]
            }
    """
    losses = []
    lrs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X, y in dataloader:
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        # Record the learning rate used during this epoch.
        lrs.append(optimizer.param_groups[0]["lr"])
        # Update the learning rate for the next epoch.
        scheduler.step()

    return {
        "losses": losses,
        "lrs": lrs,
    }