def get_chunk_util(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    n = len(tokens)
    for start_idx in range(0, n, step):
        end_idx = start_idx + chunk_size
        chunk = tokens[start_idx:end_idx]
        yield chunk 
        if end_idx >= n:
            break
    
def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks = [ ]
    for chunk in get_chunk_util(tokens, chunk_size,overlap):
        chunks.append(chunk)

    return chunks
    