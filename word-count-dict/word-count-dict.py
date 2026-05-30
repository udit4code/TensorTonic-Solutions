def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    counter_map = {}
    for sentence in sentences:
        for word in sentence:
            if word not in counter_map:
                counter_map[word] = 0
            counter_map[word] += 1
    return counter_map