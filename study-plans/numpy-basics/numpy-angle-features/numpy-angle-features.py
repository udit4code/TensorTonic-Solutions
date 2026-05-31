import numpy as np

def angle_features(angles):
    """Returns: np.ndarray of shape (3, n), rows are sin, cos, tan"""
    sine_values = np.sin(angles)
    cosine_values = np.cos(angles)
    tangent_values = np.tan(angles)
    return np.stack([sine_values, cosine_values, tangent_values])