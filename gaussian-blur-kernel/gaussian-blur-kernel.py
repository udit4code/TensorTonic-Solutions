import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list representing a 2D Gaussian filter kernel.
    """
    kernel = []
    center = size // 2
    sum_val = 0.0

    # 1. Calculate unnormalized Gaussian values
    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            val = math.exp(-(x**2 + y**2) / (2 * (sigma**2)))
            row.append(val)
            sum_val += val
        kernel.append(row)

    # 2. Normalize the kernel so the sum of all elements equals 1
    if sum_val > 0:
        for i in range(size):
            for j in range(size):
                kernel[i][j] /= sum_val

    return kernel