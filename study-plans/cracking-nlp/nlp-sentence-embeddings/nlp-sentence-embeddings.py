import numpy as np
from collections import Counter

def sentence_embeddings(sentences: list, vocab: dict, method: str = 'mean', sif_a: float = 0.001) -> list:
    """
    Pools word embeddings into one vector per sentence using mean, max, or sif.
    """
    # 1. Infer embedding width from vocab; if vocab is empty, use 1-dimensional.
    if not vocab:
        dim = 1
    else:
        sample_vec = next(iter(vocab.values()))
        dim = len(sample_vec) if hasattr(sample_vec, '__len__') else 1

    # Normalize method name
    method = method.lower()
    if method in ('weighted', 'weighted_mean'):
        method = 'sif'

    # Pre-compute word probabilities across the corpus for SIF weighting
    word_probs = {}
    if method == 'sif':
        total_tokens = 0
        token_counts = Counter()
        for sent in sentences:
            tokens = sent.split() if isinstance(sent, str) else sent
            for token in tokens:
                token_counts[token] += 1
                total_tokens += 1
        if total_tokens > 0:
            word_probs = {token: count / total_tokens for token, count in token_counts.items()}

    embeddings = []

    for sent in sentences:
        tokens = sent.split() if isinstance(sent, str) else sent
        
        # Empty sentence -> zero vector of inferred dimension
        if not tokens:
            embeddings.append(np.zeros(dim).tolist())
            continue

        # Build list of vectors for all tokens (using zero vector for unknown tokens)
        vectors = []
        for token in tokens:
            if token in vocab:
                vectors.append(np.array(vocab[token], dtype=float))
            else:
                vectors.append(np.zeros(dim))

        vectors_arr = np.array(vectors)  # Shape: (len(tokens), dim)

        if method == 'mean':
            # Averages over all token occurrences, including unknown zero vectors
            sent_vec = np.mean(vectors_arr, axis=0)

        elif method == 'max':
            # Takes the maximum independently in each coordinate, including unknown zero vectors
            sent_vec = np.max(vectors_arr, axis=0)

        elif method == 'sif':
            # Frequency-weighted mean (SIF)
            weights = np.array([sif_a / (sif_a + word_probs.get(t, 0.0)) for t in tokens])
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                sent_vec = np.sum(vectors_arr * weights[:, np.newaxis], axis=0) / sum_weights
            else:
                sent_vec = np.mean(vectors_arr, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        embeddings.append(sent_vec.tolist())

    return embeddings