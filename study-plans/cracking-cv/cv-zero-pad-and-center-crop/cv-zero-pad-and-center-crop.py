import numpy as np


def custom_zero_pad(img_arr: np.ndarray, pad_width: tuple) -> np.ndarray:
  """Pads an array with zeros given a pad_width tuple,

  replicating np.pad(..., mode="constant", constant_values=0.0) from scratch.
  """
  # 1. Calculate the new shape based on pad_width (before + size + after for each axis)
  new_shape = [
      img_arr.shape[i] + pad_width[i][0] + pad_width[i][1]
      for i in range(img_arr.ndim)
  ]

  # 2. Create a new array filled with zeros matching the original dtype
  padded = np.zeros(new_shape, dtype=img_arr.dtype)

  # 3. Create slices to place the original image into the center of the padded array
  slices = tuple(
      slice(pad_width[i][0], pad_width[i][0] + img_arr.shape[i])
      for i in range(img_arr.ndim)
  )

  # 4. Copy the original image into the designated slice
  padded[slices] = img_arr

  return padded

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
  # padded = np.pad(img_arr, pad_width, mode="constant", constant_values=0.0)
  padded = custom_zero_pad(img_arr, pad_width)

  # Calculate start and end indices for center cropping
  ph, pw = padded.shape[:2]
  start_h = (ph - crop_h) // 2
  start_w = (pw - crop_w) // 2

  # Perform the center crop using slice indexing
  cropped = padded[start_h : start_h + crop_h, start_w : start_w + crop_w]

  # Round to 4 decimals and convert back to a standard Python list
  return np.round(cropped, 4).tolist()