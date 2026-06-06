import torch
import torch.nn as nn # nn is a module

# nn.Linear(in_features, out_features) is PyTorch's implementation of a fully connected (dense) layer.
# nn.Linear(in_features, out_features)
#
# Mathematically: y = xW^T + b
#
# where:
#
#     x = input tensor
#     W = learnable weight matrix
#     b = learnable bias vector
#     y = output tensor
#
# Shapes:
#
#     x      : (batch_size, input_features)
#     W      : (hidden_features, input_features)
#     W^T    : (input_features, hidden_features)
#     b      : (hidden_features,) -> which gets broadcasted into (batch_size, hidden_features)
#
# Therefore:
#
#     xW^T   : (batch_size, hidden_features)
#     y      : (batch_size, hidden_features)
#
# The bias is automatically broadcast across the batch dimension.

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        
        # A 2-Layer MLP has 1 Linear Layer L1, followed by a ReLU Activation, followed by a second Linear Layer L2.
        # We attach these constituents into a self, so that they get registered as submodules.
        # As a result, model.parameters() will yield the weights and biases of both linear layers, and model.state_dict() will include them for saving and loading.
        
        # The type is class 'torch.nn.modules.linear.Linear
        self.linear_layer_1 = nn.Linear(in_features=self.in_features, out_features=self.hidden_size)
        # The type is class 'torch.nn.modules.activation.ReLU'
        self.relu_activation = nn.ReLU()
        # The type is class 'torch.nn.modules.linear.Linear
        self.linear_layer_2 = nn.Linear(in_features=self.hidden_size, out_features=self.out_features)
        
        

    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.relu_activation(x)
        x = self.linear_layer_2(x)
        return x