import torch.nn as nn
import torch

def train_with_early_stopping(model, train_loader, val_loader,
                              criterion, optimizer,
                              max_epochs, patience):
    """
    Returns:
        {
            'train_losses': [...],
            'val_losses': [...],
            'stopped_epoch': int
        }
    """

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        # Set to Training mode for certain layers like Dropout and BatchNorm 
        model.train()
        running_train_loss = 0.0
        
        for X, y in train_loader:
            # Why set to zero grad ? Because, PyTorch by design, does gradient accumulation. 
            optimizer.zero_grad()
            # Run Forward pass on the batch X 
            predictions = model(X)
            # Compute loss between predictions and labels y 
            loss = criterion(predictions, y)
            # Run Backward pass from loss
            loss.backward()
            # Under the hood, optimizer.step() uses no_grad() block.
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        # Validation
        # Now, set model.training_flag = False. 
        # Because, during Validation, we don't want Dropout and BatchNorm Layers to work, like it did during training. So, now, we want Dropout to not randomly deactivate activations and also, now, BatchNorm should take running average and variance. 
        model.eval()

        running_val_loss = 0.0
        # Why did we do the Validation step under a no_grad() block ? 
        with torch.no_grad():
            for X, y in val_loader:
                predictions = model(X)
                loss = criterion(predictions, y)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Now, here we put the logic of Early stopping. 
        # What do we do here ? If Validation loss has not improved (i.e grown lesser) over the last few consecutive epochs, then, we decide to stop early instead of wasting GPU compute cycles.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_epoch = epoch + 1   # 1-indexed
                break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": stopped_epoch,
    }