
def beam_search(log_probs_fn, start_token, end_token, beam_width, max_len):
    """
    Returns: list of token IDs
    """
    beams = [([start_token], 0.0)]
    complete = []

    for iteration in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == end_token:
                complete.append((seq[:-1], score))
                continue
            probs = log_probs_fn(seq)
            for token_id, log_p in enumerate(probs):
                candidates.append((seq + [token_id], score + log_p))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    all_seqs = complete + beams
    all_seqs.sort(key=lambda x: x[1], reverse=True)
    result = all_seqs[0][0]
    if result and result[-1] == end_token:
        result = result[:-1]
    return result