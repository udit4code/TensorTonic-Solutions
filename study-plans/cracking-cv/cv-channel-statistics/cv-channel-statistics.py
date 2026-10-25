import numpy as np

def channel_statistics(batch: list) -> dict:
  """Returns mean and std lists of C floats in a dictionary, rounded to four decimals."""
  X = batch 
  # Convert input list to a NumPy array
  arr = np.asarray(X, dtype=np.float64)

  # Ensure the tensor is 4-dimensional (e.g., Batch, Height, Width, Channels)
  assert arr.ndim == 4, "Input array must be 4-dimensional"

  # Compute mean and standard deviation per channel across batch and spatial axes (0, 1, 2)
  mean_vals = np.round(np.mean(arr, axis=(0, 1, 2)), 4).tolist()
  std_vals = np.round(np.std(arr, axis=(0, 1, 2)), 4).tolist()

  return {"mean": mean_vals, "std": std_vals}