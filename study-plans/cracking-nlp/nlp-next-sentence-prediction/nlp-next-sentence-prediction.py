def next_sentence_prediction(sentence_a: list, sentence_b: list, threshold: float = 0.5) -> bool:
    """
    Returns a bool indicating whether the word-overlap threshold is met.
    """
    union = set(sentence_a).union(set(sentence_b))
    intersection = set(sentence_a).intersection(set(sentence_b))
    if len(union) == 0 and len(intersection) == 0:
        return True 
    if len(union) == 0 or len(intersection) == 0:
        jaccard_similarity = 0.0
    else:
        jaccard_similarity = len(intersection) / len(union) 
    return jaccard_similarity >= threshold