import numpy as np

# Why the Numpy Implementation is faster ?
# 1. No Explicit Loops: Python loops have high overhead. 
# NumPy executes the math loops in optimized C code.
# 2. Zero-Copy Views (as_strided): In the max pooling function, 
# as_strided lets us slice windows without duplicating matrix 
# data in memory, making it scale efficiently even for large images.
def max_pool_2d(image: list, kernel_size: int, stride: int) -> list:
    """
    Returns the max-pooled 2D list of floats using NumPy stride tricks, rounded to four decimals.
    """
    # Convert input list to a 2D NumPy array of floats
    img = np.asarray(image, dtype=float)
    H, W = img.shape
    
    # Calculate the height and width of the output pooled matrix
    out_H = (H - kernel_size) // stride + 1
    out_W = (W - kernel_size) // stride + 1
    
    # Define the shape of the 4D view: (output_rows, output_cols, kernel_rows, kernel_cols)
    shape = (out_H, out_W, kernel_size, kernel_size)
    
    # Define byte strides: 
    # - Jump by 'stride' steps in memory for output coordinates
    # - Jump by 1 step in memory for internal kernel coordinates
    strides = (img.strides[0] * stride, img.strides[1] * stride, img.strides[0], img.strides[1])
    
    # Create a zero-copy, memory-efficient view of all sliding windows simultaneously
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    
    # Find the maximum value across the last two axes (the kernel patch dimensions),
    # round to 4 decimal places, and convert back to a standard nested Python list
    pooled = np.round(patches.max(axis=(-1, -2)), 4)
    return pooled.tolist()

    

def max_pool_2d_naive(image: list, kernel_size: int, stride: int) -> list:
    """
    Returns the max-pooled 2D list of floats, rounded to four decimals.
    """
    if not image or not image[0]:
        return []

    H = len(image)
    W = len(image[0])

    # Calculate output dimensions
    out_H = (H - kernel_size) // stride + 1
    out_W = (W - kernel_size) // stride + 1

    pooled_image = []

    for i in range(out_H):
        row = []
        for j in range(out_W):
            # Define window boundaries for the current patch
            r_start = i * stride
            r_end = r_start + kernel_size
            c_start = j * stride
            c_end = c_start + kernel_size

            # Find the maximum value in the window
            max_val = float('-inf')
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if image[r][c] > max_val:
                        max_val = image[r][c]

            # Append the rounded maximum value to the row
            row.append(round(float(max_val), 4))
            
        pooled_image.append(row)

    return pooled_image