import torch
import torch.nn as nn

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        """
            Returns: None
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Learnable convolution kernels.
        # Shape: (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            )
        )

        # One learnable bias per output channel.
        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )

    def forward(self, x):
        """
        Input:
            x: (batch, in_channels, H, W)

        Returns:
            (batch, out_channels, H-k+1, W-k+1)
        """

        batch_size, in_channels, H, W = x.shape
        k = self.kernel_size
        out_H = H - k + 1
        out_W = W - k + 1
        output = torch.zeros(batch_size,self.out_channels,out_H,out_W,device=x.device,dtype=x.dtype)

        # Iterate over every image.
        for b in range(batch_size):
            # Compute every output channel independently.
            for oc in range(self.out_channels):
                # Slide the kernel vertically.
                for i in range(out_H):
                    # Slide the kernel horizontally.
                    for j in range(out_W):
                        # Extract one receptive field.
                        # Shape: (in_channels, k, k)
                        patch = x[b,:,i:i+k,j:j+k]
                        # Elementwise multiplication followed by sum.
                        output[b, oc, i, j] = (patch * self.weight[oc]).sum() + self.bias[oc]

        return output
