import torch 

class TransformPipeline:
    """
        A simple image preprocessing pipeline.
        Input:
            image: Tensor of shape (H, W, C) with pixel values in [0, 255].
    
        Output:
            Float32 tensor of shape (C, H, W) with normalized pixel values.
    """

    def __init__(self, mean, std):
        # Store the per-channel mean and standard deviation.
        # view(-1, 1, 1) reshapes them from: [C] to [C, 1, 1]
        # Why ? So that they broadcast correctly across the height and width dimensions during normalization.
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image):
        # Convert the image to float32 and scale pixel values from
        # [0, 255] to [0.0, 1.0]
        # Neural networks are trained using floating-point tensors.
        # Operations such as normalization and gradient-based optimization
        # are performed in floating point.
        x = image.float() / 255.0
        # Rearrange the tensor from (H, W, C) to (C, H, W).
        # Most image-loading libraries return images in Height-Width-Channel
        # order, whereas PyTorch convolution layers expect tensors in
        # (Batch, Channels, Height, Width)
        # Therefore, each individual image must first be converted to
        # (Channels, Height, Width)
        # before it can be batched and passed to a CNN.
        x = x.permute(2, 0, 1)
        # Normalize each channel independently:
        #     x = (x - mean) / std
        # This centers the data around zero and scales it to have a
        # comparable range across channels, which typically leads to
        # faster and more stable training.
        x = (x - self.mean) / self.std
        return x
