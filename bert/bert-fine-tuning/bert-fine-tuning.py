from typing import List
import numpy as np


class MockBertEncoder:
    """Simulated BERT encoder with configurable layers."""

    def __init__(self, hidden_size: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = [
            np.random.randn(hidden_size, hidden_size) * 0.01
            for _ in range(num_layers)
        ]
        self.layer_frozen = [False] * num_layers

    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers (mark as no gradient updates)."""
        for idx in layer_indices:
            if 0 <= idx < self.num_layers:
                self.layer_frozen[idx] = True

    def unfreeze_all(self):
        """Unfreeze all layers."""
        self.layer_frozen = [False] * self.num_layers

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass: x = x @ layer_W + x for each layer (residual connection)."""
        x = embeddings
        for i, layer in enumerate(self.layers):
            x = x @ layer + x
        return x


class BertForSequenceClassification:
    """BERT with sequence classification head (uses [CLS] token)."""

    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 3):
        self.encoder = MockBertEncoder(hidden_size, num_layers)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward: encoder -> extract [CLS] (position 0) -> classifier

        Returns shape (batch, num_labels)
        """
        # Form of hidden_states: (batch, seq_len, hidden_size)
        hidden_states = self.encoder.forward(embeddings)

        # Extract the [CLS] token at position 0 from the sequence dimension
        cls_token = hidden_states[:, 0, :]

        # Linearly project into classification labels
        return cls_token @ self.classifier


class BertForTokenClassification:
    """BERT with token-level classification head (NER, POS)."""

    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 3):
        self.encoder = MockBertEncoder(hidden_size, num_layers)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward: encoder -> classify all tokens

        Returns shape (batch, seq_len, num_labels)
        """
        # Form of hidden_states: (batch, seq_len, hidden_size)
        hidden_states = self.encoder.forward(embeddings)

        # Map every single token position to class logits
        return hidden_states @ self.classifier

