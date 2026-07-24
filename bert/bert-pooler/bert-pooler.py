import numpy as np


def tanh(x):
    return np.tanh(x)


class BertPooler:
    """BERT Pooler: Extracts [CLS] and applies dense + tanh."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Extracts the first token ([CLS]) and applies linear transformation with

        tanh.

        Args:
            hidden_states: NumPy array of shape (batch, sequence_length,
              hidden_size)

        Returns:
            np.ndarray of shape (batch, hidden_size) with tanh-activated [CLS]
            output
        """
        # [CLS] token is located at the first index (index 0) of the sequence dimension
        cls_token = hidden_states[:, 0, :]
        return tanh(np.dot(cls_token, self.W) + self.b)


class SequenceClassifier:
    """Sequence classification head on top of BERT."""

    def __init__(self, hidden_size: int, num_classes: int):
        self.pooler = BertPooler(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Processes hidden states through the pooler followed by the linear

        classifier.

        Args:
            hidden_states: NumPy array of shape (batch, sequence_length,
              hidden_size)

        Returns:
            np.ndarray of shape (batch, num_classes) with classification logits
        """
        pooled_output = self.pooler.forward(hidden_states)
        return np.dot(pooled_output, self.classifier)
