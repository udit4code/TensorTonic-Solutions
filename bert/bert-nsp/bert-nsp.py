from typing import List, Tuple
import numpy as np


from typing import List, Tuple


def create_nsp_pairs(
    documents: List[List[str]], pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    """Generates Next Sentence Prediction (NSP) pairs based on dictionary specs.

    Args:
        documents: Nested list where documents[i][j] is the j-th sentence of the
          i-th document.
        pair_specs: List of dicts, each with 'doc_a', 'doc_b', 'sent_a', and
          'sent_b'.

    Returns:
        List of tuples: (sentence_A, sentence_B, is_next_label)
    """
    pairs = []

    for spec in pair_specs:
        doc_a_idx = spec["doc_a"]
        doc_b_idx = spec["doc_b"]
        sent_a_idx = spec["sent_a"]
        sent_b_idx = spec["sent_b"]

        # Extract sentences from the dataset matrix
        sentence_a = documents[doc_a_idx][sent_a_idx]
        sentence_b = documents[doc_b_idx][sent_b_idx]

        # IsNext=1 when sentences are consecutive inside the exact same document
        is_next = int(doc_a_idx == doc_b_idx and sent_b_idx == sent_a_idx + 1)

        pairs.append((sentence_a, sentence_b, is_next))

    return pairs



class NSPHead:
    """Next Sentence Prediction classification head."""

    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """Predict IsNext logits using a linear projection layer.

        Args:
            cls_hidden: NumPy array of shape (batch_size, hidden_size) or
              (hidden_size,) representing the [CLS] embedding.

        Returns:
            NumPy array of shape (batch_size, 2) or (2,) containing unnormalized logit scores.
        """
        return np.dot(cls_hidden, self.W) + self.b


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax along the last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

