# Why is validation loss used instead of training loss?

# Training loss almost always decreases because the model is optimizing directly for it. 
# Validation loss reflects performance on unseen data and is therefore a much better indicator of whether the model is still generalizing or has started to overfit. 
# Early stopping uses validation loss to halt training before overfitting becomes severe.
# We don't want to waste costly GPU compute cycles when validation loss is not improving.
# Why not stop immediately the moment we see validation loss not improving for the first time ? 
# Because, Validation loss is noisy. It is possible that validation loss hasn't reduced for last 2 epochs, but it has reduced int the next epoch. So, if we stop now before the next epoch, 
# we end up missing out on the improvement. 


import torch
import torch.nn as nn

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

    # Keep track of the best validation loss observed so far.
    best_val_loss = float("inf")

    # Counts the number of consecutive epochs with no validation improvement.
    epochs_without_improvement = 0

    # If early stopping never triggers, training is considered to have
    # completed all max_epochs.
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        # Step 1 : Training Mode
        # Put the model into training mode.
        # This mainly affects layers such as Dropout and BatchNorm:
        #   - Dropout randomly drops activations.
        #   - BatchNorm computes statistics from the current mini-batch
        #     and updates its running statistics.
        model.train()
        running_train_loss = 0.0

        for X, y in train_loader:
            # Clear gradients from the previous iteration.
            # PyTorch accumulates gradients by default, so failing to clear
            # them would cause gradients from multiple batches to be added together.
            optimizer.zero_grad()
            # Forward pass: compute predictions for the current mini-batch.
            predictions = model(X)
            # Compute the loss between predictions and ground-truth labels.
            loss = criterion(predictions, y)
            # Backpropagation: compute gradients of the loss with respect
            # to every learnable parameter.
            loss.backward()
            # Update the model parameters using the computed gradients.
            # PyTorch performs these updates inside a torch.no_grad() context
            # so that the parameter updates themselves are not tracked by autograd.
            optimizer.step()
            running_train_loss += loss.item()

        # Average training loss over all mini-batches in this epoch.
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Step 2 : Validation Mode
        
        # Switch the model to evaluation mode.
        # This changes the behaviour of certain layers:
        #   - Dropout is disabled.
        #   - BatchNorm uses the running mean and variance accumulated
        #     during training instead of statistics from the current batch.
        model.eval()

        running_val_loss = 0.0

        # During validation we only need the forward pass.
        # No gradients are required because we are not updating the model.
        # Disabling gradient tracking reduces both memory usage and runtime.
        with torch.no_grad():
            for X, y in val_loader:
                predictions = model(X)
                loss = criterion(predictions, y)
                running_val_loss += loss.item()

        # Average validation loss over the validation set.
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step 3 : Early Stopping
        # If validation loss improves, remember the new best value and
        # reset the patience counter.
        if avg_val_loss < best_val_loss:
            # Also, in production, we also save the best checkpoint whenever validation improves.
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

        else:
            # Validation loss did not improve this epoch.
            epochs_without_improvement += 1

            # Stop training once the model has failed to improve for
            # 'patience' consecutive epochs.
            if epochs_without_improvement >= patience:

                stopped_epoch = epoch + 1      # Return a 1-indexed epoch.
                break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": stopped_epoch,
    }