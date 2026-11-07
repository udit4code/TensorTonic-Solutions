import torch

def patchify_util(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Returns a (B, N, P * P * C) tensor using float64 to prevent precision loss for large numbers.
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0, f"patch_size {patch_size} does not divide H {H}"
    assert W % patch_size == 0, f"patch_size {patch_size} does not divide W {W}"

    h, w = H // patch_size, W // patch_size
    
    # Step 1 : Reshape H and W into spatial grid dimensions and patch dimensions
    x = images.reshape(B, C, h, patch_size, w, patch_size)

    # Step 2 : Permute to order patch internals as (local_row, local_column, channel)
    x = x.permute(0, 2, 4, 3, 5, 1)
    
    # Step 3 : Flatten spatial grid and patch dimensions
    x = x.reshape(B, h * w, patch_size * patch_size * C)
    
    # Step 4 : Round tensor values to 4 decimal places
    x = torch.round(x, decimals=4)
    
    return x
    
def patchify(images: list, patch_size: int) -> list:
    """
    Returns a nested float list of shape (B, num_patches, patch_size * patch_size * C),
    rounded to four decimals, using float64 to preserve precision for large numbers.
    """
    # Load into torch using float64 (double precision) to support large numbers > 16,777,216
    images_tensor = torch.tensor(images, dtype=torch.float64)
    
    # Process using the utility tensor function
    output_tensor = patchify_util(images_tensor, patch_size)
    
    # Convert the resulting tensor back to a standard Python list
    return output_tensor.tolist()