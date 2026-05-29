import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        # YOUR CODE HERE
        batch_size, seq_len = token_ids.shape
        # Step 1 : Get Token embeddings lookup
        token_embeds = self.token_embeddings[token_ids]
        # shape: (batch, seq_len, hidden_size)

        # Step 2 : Get Position embeddings lookup
        position_ids = np.arange(seq_len)
        # shape: (seq_len,)
        position_embeds = self.position_embeddings[position_ids]
        # shape: (seq_len, hidden_size)
        # Broadcast positions across batch: (seq_len, hidden) becomes (1, seq_len, hidden)
        position_embeds = position_embeds[np.newaxis, :, :]
        # Step 3: Get Segment embeddings lookup
        segment_embeds = self.segment_embeddings[segment_ids]
        # shape: (batch, seq_len, hidden_size)
        # Step 4 : Get Final BERT embedding
        embeddings = (
            token_embeds
            + position_embeds
            + segment_embeds
        )
        return embeddings
