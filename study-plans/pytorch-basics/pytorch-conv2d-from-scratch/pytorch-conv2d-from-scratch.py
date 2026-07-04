import torch
import torch.nn as nn

class Conv2d_naive(nn.Module):

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


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        """
            A 2D convolution implemented using im2col + matrix multiplication.
    
            Input:
                (batch, in_channels, H, W)
    
            Output:
                (batch, out_channels, H-k+1, W-k+1)
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
        # One learnable bias for each output channel.
        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )

    def forward(self, x):
        batch_size, _, H, W = x.shape
        k = self.kernel_size
        out_H = H - k + 1
        out_W = W - k + 1
        # Step 1 : Extract every k×k sliding window from the input.
        # After unfolding: (B,C,H,W) becomes (B,C,out_H,out_W,k,k)
        patches = x.unfold(2, k, 1).unfold(3, k, 1)
        # Step 2 : Rearrange the tensor so that every sliding window becomes one row.
        # Shape: (B,out_H,out_W,C,k,k)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        # Step 3 : Flatten every receptive field.
        # Example: 3×3×3 becomes length 27
        # Shape: (B, out_H*out_W, C*k*k)
        patches = patches.reshape(
            batch_size,
            out_H * out_W,
            -1
        )
        # Step 4 : Flatten every convolution kernel.
        # Shape: (out_channels,C*k*k)
        weight = self.weight.reshape(self.out_channels,-1)
        # Step 5 :
        # Compute every convolution using one matrix multiplication.
        # Instead of thousands of patch × kernel operations, perform one batched GEMM. 
        # Result: (B,out_H*out_W,out_channels)
        output = patches @ weight.T
        # Step 6 :
        # Add one bias for every output channel.
        # Broadcasting automatically expands the bias across all spatial locations.
        output += self.bias
        # Step 7 :
        # Reshape back into the expected CNN output format.
        # (B, out_H*out_W, out_channels) -> (B, out_channels, out_H, out_W)
        output = output.reshape(
            batch_size,
            out_H,
            out_W,
            self.out_channels
        )
        output = output.permute(0, 3, 1, 2)
        return output