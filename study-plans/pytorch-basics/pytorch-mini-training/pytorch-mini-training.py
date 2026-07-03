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
        # Step 1 : Initialise the gradients of the optimizer to zero. 
        # Why ? So that old gradients are cleared. 
        optimizer.zero_grad()
        # Step 2 : Compute Forward Pass, by passing the model through the inputs
        prediction_outputs = model(inputs)
        # Step 3 : Compute the losee between the predictions and the target labels
        loss = criterion(prediction_outputs, targets)
        # Step 4 : Do the Backward pass
        loss.backward()
        # tep 5 : Update the parameters via the optimizer 
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
