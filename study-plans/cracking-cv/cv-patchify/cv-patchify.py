import torch

def patchify_util(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Returns a (B, N, C * P * P) tensor: every patch of every image, flattened, 
    with pixels ordered by row, then column, then channel (channel changing fastest).
    """
    # BCHW: Batch, Channels (e.g., 3 for RGB), Height, Width (each image is a C × H × W tensor).
    # So, images is a batch of B images, where each image is a 3d tensor. 
    # 1 image consists of C channels and each channel, within an image, maps to a H x W matrix.
    B, C, H, W = images.shape
    assert H % patch_size == 0, f"patch_size {patch_size} does not divide H {H}"
    assert W % patch_size == 0, f"patch_size {patch_size} does not divide W {W}"

    # h and w represent the number of patches that fit vertically and horizontally across the image's height H and width W.
    h, w = H // patch_size, W // patch_size
    
    # Step 1 : Reshape H and W into spatial grid dimensions (h, w) and patch dimensions (patch_size, patch_size)
    # Dimension evolution: (B, C, H, W) -> (B, C, h, patch_size, w, patch_size)
    x = images.reshape(B, C, h, patch_size, w, patch_size)

    # Step 2 : Permute to order patch internals as (local_row, local_column, channel) 
    # where channel changes fastest, then column, then row.
    # Dimension evolution: (B, C, h, patch_size, w, patch_size) -> (B, h, w, patch_size, patch_size, C)
    # Permutation mapping: (0:B, 2:h, 4:w, 3:patch_row, 5:patch_col, 1:C) -> (0, 2, 4, 3, 5, 1)
    x = x.permute(0, 2, 4, 3, 5, 1)
    
    # Step 3 : Flatten the spatial grid into N (h * w) and the patch/channel dimensions into patch_size * patch_size * C
    # Dimension evolution: (B, h, w, patch_size, patch_size, C) -> (B, h * w, patch_size * patch_size * C)
    # So, overall, we have now h * w vectors per image, and each patch is flattened into a 1D vector of length patch_size * patch_size * C.
    x = x.reshape(B, h * w, patch_size * patch_size * C)
    
    # Step 4 : Round tensor values to 4 decimal places as requested by the prompt
    x = torch.round(x, decimals=4)
    
    return x
    
def patchify(images: list, patch_size: int) -> list:
    """
    Returns a nested float list of shape (B, num_patches, patch_size * patch_size * C),
    rounded to four decimals. Uses float64 to prevent precision loss for large numbers (> 16,777,216).
    """
    # Convert input list to a PyTorch tensor using float64 (double precision) to support large numbers without precision truncation
    images_tensor = torch.tensor(images, dtype=torch.float64)
    
    # Process using the utility tensor function
    output_tensor = patchify_util(images_tensor, patch_size)
    
    # Convert the resulting tensor back to a standard Python list
    return output_tensor.tolist()