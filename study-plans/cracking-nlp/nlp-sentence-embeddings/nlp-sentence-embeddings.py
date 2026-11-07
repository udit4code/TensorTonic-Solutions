import numpy as np


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


# Q: How did we calculate sentence embeddings, before using static word vectors ? 
# Ans : In the context of pooling-based approaches (especially with static word vectors like Word2Vec, 
# GloVe, or FastText), a sentence embedding is indeed formed by taking the individual vector representations 
# of its words and aggregating them using a mathematical function (like mean, max, or SIF weighting). 
# We treat a sentence as a "bag (multiset) of vectors," where word order doesn't strictly matter 
# to the pooling operation itself, and everything is reduced to a single vector through averaging, taking maximums, or frequency weighting. A bag (or multiset) preserves frequencies. 
# Because our implementation loops through every token occurrence in the sentence (for token in tokens), 
# repeating a word means its vector is included multiple times, 
# which properly pulls the mean or sum in that direction.
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
    # Step 1 : Infer dimension or default to 1 if vocab is empty
    dim = get_embedding_dimension(vocab)
    
    # Step 2 : Normalize method name
    method = method.lower()
    if method in ('weighted', 'weighted_mean'):
        method = 'sif'
        
    # Step 3 : Pre-compute corpus word probabilities if SIF pooling is requested
    word_probs = compute_corpus_probabilities(sentences) if method == 'sif' else {}
    embeddings = []
    
    # Step 4 : Process each sentence individually
    for sent in sentences:
        tokens = sent.split() if isinstance(sent, str) else sent
        # Step 4.1 : Handle empty sentences with a zero vector
        if not tokens:
            embeddings.append(np.zeros(dim).tolist())
            continue  
        # Step 4.2 Collect vectors for all tokens in the sentence (using zero vectors for OOV tokens)
        vectors = [get_token_vector(token, vocab, dim) for token in tokens]
        vectors_arr = np.array(vectors)  # Shape: (num_tokens, dim)
        # Step 4.3 : Apply the chosen pooling method
        sent_vec = pool_vectors(vectors_arr, tokens, method, word_probs, sif_a)
        # Step 4.4 : Convert NumPy array back to a Python list
        embeddings.append(sent_vec.tolist())
        
    return embeddings