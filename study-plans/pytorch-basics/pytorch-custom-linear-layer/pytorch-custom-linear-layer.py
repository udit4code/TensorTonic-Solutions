import torch
import torch.nn as nn

# torch.empty returns a tensor filled with uninitialized data of a specified shape. 

# Because the allocated memory block is not zeroed out or filled with a specific default value, it contains whatever arbitrary bits happened to reside in that memory address beforehand (often referred to as garbage values)

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        # A Customised Linear Layer has 2 types of learnable parameters -> weights matrix and bias vector.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initially, they are filled with garbage values. So, we have to initialise them. 
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        # So each row of x is multiplied by W^T and the bias is added. 
        return x @ self.weight.T + self.bias
        
