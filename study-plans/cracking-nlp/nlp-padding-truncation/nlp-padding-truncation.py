def pad_and_truncate(sequences: list, max_length: int, pad_value: int = 0) -> list:
    """
    Returns one integer list of length max_length per input sequence.
    """
    result = []
    for seq in sequences:
        if len(seq) >= max_length:
            result.append(seq[:max_length])
        else:
            result.append(seq + [pad_value] * (max_length - len(seq)))
    return result