import numpy as np

import numpy as np


class Dropout:
    def __init__(self, p: float = 0.5):
        """
        Initializes the Dropout layer with a drop probability p.
        """
        self.p = p

    def __call__(
        self,
        x: list | np.ndarray,
        training: bool = True,
        rng: np.random.Generator = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies inverted dropout to the input x.
        
        Args:
            x: Input list or NumPy array.
            training: If True, applies dropout. If False, acts as an identity pass-through.
            rng: Optional NumPy random generator for reproducibility.
            
        Returns:
            tuple: (output, dropout_pattern) as NumPy arrays matching the shape of x.
        """
        # Convert input to a NumPy array
        arr = np.asarray(x, dtype=float)
        
        # During evaluation mode or if p is 0, bypass dropout (identity pass-through)
        if not training or self.p == 0.0:
            dropout_pattern = np.ones_like(arr)
            return arr, dropout_pattern
            
        # Initialize default generator if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Edge case: if drop probability is 1.0, everything is zeroed out
        if self.p >= 1.0:
            dropout_pattern = np.zeros_like(arr)
            output = np.zeros_like(arr)
            return output, dropout_pattern
            
        # Inverted dropout: scale active units by 1 / (1 - p)
        scale = 1.0 / (1.0 - self.p)
        dropout_pattern = (rng.random(arr.shape) >= self.p).astype(float) * scale
        
        # Apply mask to input array
        output = arr * dropout_pattern
        
        return output, dropout_pattern
        

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Instantiate the Dropout layer and invoke it
    layer = Dropout(p=p)
    return layer(x, training=True, rng=rng)