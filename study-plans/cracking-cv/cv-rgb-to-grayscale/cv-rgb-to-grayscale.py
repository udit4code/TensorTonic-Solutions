import numpy as np 

# Using naive python
def rgb_to_grayscale_naive(image: list) -> list:
    """
    Returns the H-by-W grayscale list of floats, rounded to four decimals.
    """
    transform = lambda r, g, b : 0.299*r + 0.587*g + 0.114*b 
    return [[round(transform(r, g, b), 4) for r, g, b in row] for row in image]

# Using dot product
def rgb_to_grayscale(image: list) -> list:
  """Returns the H-by-W grayscale list of floats, rounded to four decimals,

  optimised using NumPy vectorisation.
  """
  # Convert input list to a NumPy array (avoids explicit Python loops)
  img_arr = np.asarray(image, dtype=np.float64)

  # Define the weights for R, G, B
  weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)

  # Vectorised dot product across the last axis (the RGB channel dimension)
  gray_arr = np.dot(img_arr, weights)

  # Round to 4 decimals and convert back to a standard Python nested list
  return np.round(gray_arr, 4).tolist()