import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vocab_word_to_index_map = {}
    for index, word in enumerate(vocab):
        vocab_word_to_index_map[word] = index
    bow_vector = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in vocab_word_to_index_map:
            bow_vector[vocab_word_to_index_map.get(token)] += 1
    return bow_vector