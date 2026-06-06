import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Returns: None
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        # torch.rand_like(x) is a PyTorch function that returns a new tensor filled with random numbers from a uniform distribution on the interval \([0, 1)\), matching the exact shape, data type (dtype), and device of the input tensor 
        mask = (torch.rand(x.shape) >= self.p).to(torch.float32)
        return x * mask / (1 - self.p)
