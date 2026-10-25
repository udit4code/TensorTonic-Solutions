import numpy as np


def pad_and_center_crop(
    image: list, pad: int, crop_h: int, crop_w: int
) -> list:
  """Returns the crop_h-by-crop_w list of floats, rounded to four decimals,

  using NumPy for padding and center cropping.
  """
  img_arr = np.asarray(image, dtype=np.float64)

  # Define padding width for spatial dimensions (H, W), leaving channels untouched if 3D
  if img_arr.ndim == 2:
    pad_width = ((pad, pad), (pad, pad))
  elif img_arr.ndim == 3:
    pad_width = ((pad, pad), (pad, pad), (0, 0))
  else:
    raise ValueError("Image must be 2D (H, W) or 3D (H, W, C).")

  # Pad the image with zeros (constant padding)
  padded = np.pad(img_arr, pad_width, mode="constant", constant_values=0.0)

  # Calculate start and end indices for center cropping
  ph, pw = padded.shape[:2]
  start_h = (ph - crop_h) // 2
  start_w = (pw - crop_w) // 2

  # Perform the center crop using slice indexing
  cropped = padded[start_h : start_h + crop_h, start_w : start_w + crop_w]

  # Round to 4 decimals and convert back to a standard Python list
  return np.round(cropped, 4).tolist()