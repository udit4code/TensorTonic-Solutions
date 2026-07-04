import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
            A basic ResNet residual block.
            Input: (batch, channels, H, W)
            Output: (batch, channels, H, W)
        """
        super().__init__()

        # First 3x3 convolution.
        # padding=1 ensures that the spatial dimensions (height and width) remain unchanged.
        # Input: (B, C, H, W)
        # Output: (B, C, H, W)
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1)
        # Batch Normalization after the first convolution.
        self.bn1 = nn.BatchNorm2d(channels)
        # Second 3x3 convolution.
        # Again, padding=1 preserves the spatial dimensions so that the residual (skip) connection can be added elementwise.
        self.conv2 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1)
        # Batch Normalization after the second convolution.
        self.bn2 = nn.BatchNorm2d(channels)
        # ReLU activation :  A single ReLU module can be reused multiple times because it has no learnable parameters.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            Args:
                x: Shape: (batch, channels, H, W)
            Returns:
                Tensor of the same shape.
        """

        # Preserve the original input.
        # This is the identity (skip) connection that will later be added back to the output of the convolutional branch.
        identity = x
        # First convolution block 
        # Conv -> BatchNorm -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Second convolution block.
        # Conv -> BatchNorm
        # Notice there is NO ReLU here yet.
        out = self.conv2(out)
        out = self.bn2(out)
        # Residual (skip) connection.
        # Add the original input elementwise to the output of the convolutional branch.
        # This is only possible because both tensors have exactly the same shape.
        out = out + identity
        # Apply the final ReLU activation.
        out = self.relu(out)
        return out