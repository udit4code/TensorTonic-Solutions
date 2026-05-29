import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    # YOUR CODE HERE
    masked_ids = token_ids.copy()
    # Step 1 : Create labels
    labels = np.full(
        token_ids.shape,
        -100
    )
    # only masked positions have targets
    labels[mask_positions] = token_ids[mask_positions]
    # Step 2: Apply 80-10-10 rule
    # 80% : [MASK]
    mask_case = (
        mask_positions &
        (replace_probs < 0.8)
    )
    masked_ids[mask_case] = mask_token_id
    # 10% : random token
    random_case = (
        mask_positions &
        (replace_probs >= 0.8) &
        (replace_probs < 0.9)
    )
    masked_ids[random_case] = random_tokens[random_case]
    # remaining 10% : keep unchanged , no operation needed
    return masked_ids, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        # YOUR CODE HERE
        logits = hidden_states @ self.W + self.b
        return logits
