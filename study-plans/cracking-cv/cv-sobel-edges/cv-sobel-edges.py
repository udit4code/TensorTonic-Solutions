import numpy as np


def sobel_edges(image: list) -> dict:
    """Returns gx, gy, and magnitude as H-by-W float lists,
       rounded to four decimals, computed using NumPy convolution.
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape
    
    # Step 1 : Define standard Sobel kernels for horizontal (Gx) and vertical (Gy) gradients
    Kx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    # Step 2 : Pad the image with zeros (1 pixel border for 3x3 kernel)
    padded = np.pad(img, pad_width=1, mode="constant", constant_values=0.0)
    
    # Step 3 : Perform 2D convolution efficiently using NumPy slices
    gx = np.zeros((H, W), dtype=np.float64)
    gy = np.zeros((H, W), dtype=np.float64)
    
        
    for i in range(3):
        for j in range(3):
            window = padded[i : i + H, j : j + W]
            gx += window * Kx[i, j]
            gy += window * Ky[i, j]
    
    # Step 4 : Compute gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Step 5 : Round to 4 decimals and convert back to standard Python lists
    return {
        "gx": np.round(gx, 4).tolist(),
        "gy": np.round(gy, 4).tolist(),
        "magnitude": np.round(magnitude, 4).tolist(),
    }