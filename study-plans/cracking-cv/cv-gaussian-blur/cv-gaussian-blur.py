import numpy as np


def _create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
  """Generates and normalizes a 2D Gaussian kernel using separable 1D vectors."""
  k = int(kernel_size)
  c = (k - 1) / 2.0
  idx = np.arange(k, dtype=np.float64)
  g_1d = np.exp(-0.5 * ((idx - c) / sigma) ** 2)

  # Construct 2D kernel via outer product and normalize
  G_2d = np.outer(g_1d, g_1d)
  return G_2d / np.sum(G_2d)


def _pad_image(img: np.ndarray, kernel_size: int) -> np.ndarray:
  """Pads the image with zeros using zero-allocation arrays."""
  H, W = img.shape
  k = int(kernel_size)
  p = k // 2

  padded = np.zeros((H + 2 * p, W + 2 * p), dtype=np.float64)
  padded[p : p + H, p : p + W] = img
  return padded


def _convolve_2d(padded_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  """Performs pixel-centric 2D convolution by sliding a local window."""
  k = kernel.shape[0]
  H = padded_img.shape[0] - k + 1
  W = padded_img.shape[1] - k + 1

  out = np.zeros((H, W), dtype=np.float64)
  for i in range(H):
    for j in range(W):
      window = padded_img[i : i + k, j : j + k]
      out[i, j] = np.sum(window * kernel)

  return out


def gaussian_blur_2d(image: list, kernel_size: int, sigma: float) -> list:
  """Returns the H-by-W blurred list of floats, rounded to four decimals,

  orchestrated via modular helper functions.
  """
  img = np.asarray(image, dtype=np.float64)

  # 1. Generate kernel
  kernel = _create_gaussian_kernel(kernel_size, sigma)

  # 2. Pad image borders
  padded_img = _pad_image(img, kernel_size)

  # 3. Apply convolution
  blurred = _convolve_2d(padded_img, kernel)

  # 4. Round and format output
  return np.round(blurred, 4).tolist()