import numpy as np
from collections import Counter

def get_embedding_dimension(vocab: dict) -> int:
    """
    Infers the embedding width from the vocabulary.
    Defaults to 1 if the vocabulary is empty.
    """
    if not vocab:
        return 1
    
    # Get the first vector from the vocabulary to check its length/shape
    sample_vec = next(iter(vocab.values()))
    if hasattr(sample_vec, '__len__'):
        return len(sample_vec)
    return 1


def compute_corpus_probabilities(sentences: list) -> dict:
    """
    Computes token probabilities p(w) across the entire corpus 
    for use in SIF (Smooth Inverse Frequency) weighting.
    """
    total_tokens = 0
    token_counts = {}
    
    for sent in sentences:
        # Split string into tokens if sentence is a string; otherwise use as-is
        tokens = sent.split() if isinstance(sent, str) else sent
        for token in tokens:
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1
            total_tokens += 1
            
    # Calculate relative frequency for each token
    if total_tokens == 0:
        return {}
        
    return {token: count / total_tokens for token, count in token_counts.items()}


def get_token_vector(token: str, vocab: dict, dim: int) -> np.ndarray:
    """
    Retrieves the embedding vector for a token.
    Returns a zero vector of size 'dim' if the token is absent from vocab.
    """
    if token in vocab:
        return np.array(vocab[token], dtype=float)
    return np.zeros(dim)


def pool_vectors(vectors_arr: np.ndarray, tokens: list, method: str, word_probs: dict, sif_a: float) -> np.ndarray:
    """
    Pools a collection of word vectors into a single sentence vector 
    using the specified method ('mean', 'max', or 'sif').
    """
    if method == 'mean':
        # Average over all token occurrences (including zero vectors for unknown tokens)
        return np.mean(vectors_arr, axis=0)
        
    elif method == 'max':
        # Take the maximum independently across each coordinate
        return np.max(vectors_arr, axis=0)
        
    elif method in ('sif', 'weighted', 'weighted_mean'):
        # Calculate SIF weights: w(w) = a / (a + p(w))
        weights = np.array([sif_a / (sif_a + word_probs.get(t, 0.0)) for t in tokens])
        sum_weights = np.sum(weights)
        
        # Weighted average or fallback to mean if sum of weights is zero
        if sum_weights > 0:
            return np.sum(vectors_arr * weights[:, np.newaxis], axis=0) / sum_weights
        else:
            return np.mean(vectors_arr, axis=0)
            
    else:
        raise ValueError(f"Unknown pooling method: '{method}'. Choose from 'mean', 'max', or 'sif'.")


def sentence_embeddings(sentences: list, vocab: dict, method: str = 'mean', sif_a: float = 0.001) -> list:
    """
    Pools supplied word embeddings into one vector per sentence using mean, max, or sif.
    
    Parameters:
    - sentences: List of sentences (either raw strings or pre-tokenized lists of strings)
    - vocab: Dictionary mapping words to their vector representations
    - method: Pooling strategy ('mean', 'max', or 'sif')
    - sif_a: Smoothing parameter for SIF frequency-weighted mean
    
    Returns:
    - A list of float lists, where each list represents a sentence embedding.
    """
    # 1. Infer dimension or default to 1 if vocab is empty
    dim = get_embedding_dimension(vocab)
    
    # 2. Normalize method name
    method = method.lower()
    if method in ('weighted', 'weighted_mean'):
        method = 'sif'
        
    # 3. Pre-compute corpus word probabilities if SIF pooling is requested
    word_probs = compute_corpus_probabilities(sentences) if method == 'sif' else {}
    
    embeddings = []
    
    # 4. Process each sentence individually
    for sent in sentences:
        tokens = sent.split() if isinstance(sent, str) else sent
        
        # Handle empty sentences with a zero vector
        if not tokens:
            embeddings.append(np.zeros(dim).tolist())
            continue
            
        # Collect vectors for all tokens in the sentence (using zero vectors for OOV tokens)
        vectors = [get_token_vector(token, vocab, dim) for token in tokens]
        vectors_arr = np.array(vectors)  # Shape: (num_tokens, dim)
        
        # Apply the chosen pooling method
        sent_vec = pool_vectors(vectors_arr, tokens, method, word_probs, sif_a)
        
        # Convert NumPy array back to a Python list
        embeddings.append(sent_vec.tolist())
        
    return embeddings