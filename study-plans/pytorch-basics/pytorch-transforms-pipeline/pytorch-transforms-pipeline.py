import torch

class TransformPipeline:
    """
    Returns: float32 tensor of shape (C, H, W) from __call__
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image):
        # Why float ? Because PyTorch Model works with floats, instead of unsigned integers from 0 to 255. 
        # Integer arithmetic is not differentiable. 
        x = image.float() / 255.0 
        # Go from H,W,C to C,H,W. Why ? 
        # Because OpenCV stores images in disk in HWC format. 
        # When we load that image from disk, we have to switch it to CHW before feeding to PyTorch Model. 
        # Why ? Because, CHW is ideal when it comes to computing convolutions. 
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / (self.std)
        return x 
