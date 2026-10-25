import numpy as np


def normalize_image(image: list, mean: list, std: list) -> list:
  """Returns the normalized H-by-W-by-C list of floats, rounded to four decimals,

  using NumPy for array broadcasting and Python's native round() for exact precision.
  """
  # Convert inputs to NumPy arrays
  img_arr = np.asarray(image, dtype=np.float64)
  mean_arr = np.asarray(mean, dtype=np.float64)
  std_arr = np.asarray(std, dtype=np.float64)

  # Perform vectorised normalization using NumPy broadcasting
  norm_arr = (img_arr - mean_arr) / std_arr

  # Use Python's built-in round() element-wise to match the expected test precision
  return [
      [[round(float(val), 4) for val in pixel] for pixel in row]
      for row in norm_arr
  ]