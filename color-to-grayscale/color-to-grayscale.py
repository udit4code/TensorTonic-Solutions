import numpy as np 

def color_to_grayscale(image: list) -> list:
    """
    Returns the luminance value of every RGB pixel.
    """
    # Step 1 : Convert input list to a NumPy array (avoids explicit Python loops)
    img_arr = np.asarray(image, dtype=np.float64)

    # Step 2 : Define the weights for R, G, B
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    
    # Step 3 : Vectorised dot product across the last axis (the RGB channel dimension)
    gray_arr = np.dot(img_arr, weights)
    
    # Step 4 : Round to 4 decimals and convert back to a standard Python nested list
    return np.round(gray_arr, 4).tolist()
    