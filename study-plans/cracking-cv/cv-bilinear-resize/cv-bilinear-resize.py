import numpy as np


def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
  """Returns the new_h-by-new_w resized list of floats, rounded to four decimals,

  using explicit for-loops and corner-aligned coordinate mapping.
  """
  img = np.asarray(image, dtype=np.float64)
  is_3d = img.ndim == 3

  # Step 1 : Add a dummy channel dimension if 2D for uniform processing
  if not is_3d:
    img = img[..., np.newaxis]

  H, W, C = img.shape

  # Step 2 : Initialize an empty output array for the resized image
  res = np.zeros((new_h, new_w, C), dtype=np.float64)

  # Step 3 : Iterate through every pixel in the new target image
  for i in range(new_h):
    for j in range(new_w):
      # Step 3.1 : Corner-aligned coordinate mapping back to the source image
      y = i * (H - 1) / (new_h - 1) if new_h > 1 else 0.0
      x = j * (W - 1) / (new_w - 1) if new_w > 1 else 0.0

      # Step 3.2 : Find integer bounding box indices (top-left neighbors)
      y0 = int(np.floor(y))
      x0 = int(np.floor(x))

      # Step 3.3 : Find bottom-right neighbors with boundary clipping
      y1 = min(y0 + 1, H - 1)
      x1 = min(x0 + 1, W - 1)
      y0 = max(min(y0, H - 1), 0)
      x0 = max(min(x0, W - 1), 0)

      # Step 3.4 : Calculate fractional weights for interpolation
      dy = y - y0
      dx = x - x0

      # Step 3.5 : Extract the four surrounding pixel values
      Ia = img[y0, x0]  # Top-left
      Ib = img[y1, x0]  # Bottom-left
      Ic = img[y0, x1]  # Top-right
      Id = img[y1, x1]  # Bottom-right

      # Step 3.6 : Apply bilinear interpolation formula
      top = Ia * (1 - dx) + Ic * dx
      bottom = Ib * (1 - dx) + Id * dx
      res[i, j] = top * (1 - dy) + bottom * dy

  # Step 4 : Strip dummy channel dimension if input was originally 2D
  if not is_3d:
    res = res[..., 0]

  return np.round(res, 4).tolist()
    
def bilinear_resize_using_meshgrid_api(image: list, new_h: int, new_w: int) -> list:
  """Returns the new_h-by-new_w resized list of floats, rounded to four decimals,

  using standard corner-aligned coordinate mapping.
  """
  img = np.asarray(image, dtype=np.float64)
  is_3d = img.ndim == 3

  if not is_3d:
    img = img[..., np.newaxis]

  H, W, C = img.shape

  # Step 1 : Generate target coordinate grids using (H - 1) / (new_h - 1) alignment
  y_indices = np.arange(new_h, dtype=np.float64)
  x_indices = np.arange(new_w, dtype=np.float64)

  y_scaled = (
      y_indices * (H - 1) / (new_h - 1)
      if new_h > 1
      else np.zeros(new_h, dtype=np.float64)
  )
  x_scaled = (
      x_indices * (W - 1) / (new_w - 1)
      if new_w > 1
      else np.zeros(new_w, dtype=np.float64)
  )

  # Step 2 : Compute integer and fractional parts for interpolation
  y0 = np.floor(y_scaled).astype(int)
  x0 = np.floor(x_scaled).astype(int)
  y1 = np.clip(y0 + 1, 0, H - 1)
  x1 = np.clip(x0 + 1, 0, W - 1)
  y0 = np.clip(y0, 0, H - 1)
  x0 = np.clip(x0, 0, W - 1)

  dy = y_scaled - y0
  dx = x_scaled - x0

  # Step 3 : Create 2D coordinate grids and weight grids
  y0_grid, x0_grid = np.meshgrid(y0, x0, indexing="ij")
  y1_grid, x1_grid = np.meshgrid(y1, x1, indexing="ij")

  dy_grid, _ = np.meshgrid(dy, np.zeros(new_w), indexing="ij")
  _, dx_grid = np.meshgrid(np.zeros(new_h), dx, indexing="ij")

  dy_grid = dy_grid[..., np.newaxis]
  dx_grid = dx_grid[..., np.newaxis]

  # Step 4 : Extract the four surrounding pixel values
  Ia = img[y0_grid, x0_grid]
  Ib = img[y1_grid, x0_grid]
  Ic = img[y0_grid, x1_grid]
  Id = img[y1_grid, x1_grid]

  # Step 5 : Apply bilinear interpolation formula
  top = Ia * (1 - dx_grid) + Ic * dx_grid
  bottom = Ib * (1 - dx_grid) + Id * dx_grid
  res = top * (1 - dy_grid) + bottom * dy_grid

  if not is_3d:
    res = res[..., 0]

  return np.round(res, 4).tolist()