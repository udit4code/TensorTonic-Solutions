def max_pool_2d(image: list, kernel_size: int, stride: int) -> list:
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