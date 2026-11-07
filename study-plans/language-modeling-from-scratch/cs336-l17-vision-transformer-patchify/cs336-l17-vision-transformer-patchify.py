import torch


def vision_transformer_patchify(
    images: torch.Tensor, patch_height: int, patch_width: int,
) -> dict:
    """
    Splits a channel-first image batch (B, C, H, W) into non-overlapping rectangular patches.
    Returns a dictionary of PyTorch tensors:
      - 'tokens': shape (batch, patch_count, channels * patch_height * patch_width) preserving image dtype.
      - 'coordinates': shape (patch_count, 2) with int64 dtype containing [grid row, grid column].
    Both outputs stay on the image device.
    """
    # BCHW: Batch, Channels (e.g., 3 for RGB), Height, Width (each image is a C × H × W tensor).
    B, C, H, W = images.shape
    device = images.device

    # Ensure patch dimensions divide the image height and width evenly
    assert H % patch_height == 0, f"patch_height {patch_height} does not divide H {H}"
    assert W % patch_width == 0, f"patch_width {patch_width} does not divide W {W}"

    # h and w represent the number of patches that fit vertically and horizontally
    h = H // patch_height
    w = W // patch_width
    patch_count = h * w  # Total number of patches per image (N)

    # Step 1 : Reshape H and W into spatial grid dimensions (h, w) and local patch dimensions (patch_height, patch_width)
    # Shape evolution: (B, C, H, W) -> (B, C, h, patch_height, w, patch_width)
    x = images.reshape(B, C, h, patch_height, w, patch_width)

    # Step 2 : Permute to bring the spatial grid (h, w) before channels and patch pixels
    # Shape evolution: (B, C, h, patch_height, w, patch_width) -> (B, h, w, C, patch_height, patch_width)
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Step 3 : Flatten the spatial grid into N (h * w) and patch/channel dimensions into C * patch_height * patch_width
    # Shape evolution: (B, h, w, C, patch_height, patch_width) -> (B, patch_count, C * patch_height * patch_width)
    # Preserves the image dtype and stays on the same device.
    tokens = x.reshape(B, patch_count, C * patch_height * patch_width)

    # Step 4 : Generate zero-based patch-grid coordinates [grid row, grid column] in row-first order without meshgrid
    # Shape evolution: create flat indices from 0 to patch_count - 1 -> (patch_count,)
    flat_indices = torch.arange(patch_count, dtype=torch.int64, device=device)
    
    # Compute row and column indices using integer division and modulo
    # rows = flat_indices // w (which row the patch belongs to)
    # cols = flat_indices % w (which column the patch belongs to)
    rows = flat_indices // w
    cols = flat_indices % w
    
    # Stack into shape (patch_count, 2) with int64 dtype on the image device
    # Shape evolution: (patch_count,), (patch_count,) -> (patch_count, 2)
    coordinates = torch.stack([rows, cols], dim=-1)

    return {
        "tokens": tokens,
        "coordinates": coordinates
    }