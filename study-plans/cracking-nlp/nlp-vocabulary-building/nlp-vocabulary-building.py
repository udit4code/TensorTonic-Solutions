def tokenize_corpus(corpus: list) -> list:
    """
    Tokenizes each sentence in the corpus if it's a raw string, 
    or preserves it as-is if it's already a list of tokens.
    """
    return [sent.split() if isinstance(sent, str) else sent for sent in corpus]


def count_tokens(tokenized_corpus: list) -> Counter:
    """
    Counts occurrences of each token across the entire tokenized corpus.
    Excludes reserved tokens so they don't compete for regular ranking slots.
    """
    token_counts = { }
    for tokens in tokenized_corpus:
        for token in tokens:
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1
            
    # Remove reserved tokens from counts if they accidentally appeared
    token_counts.pop("<pad>", None)
    token_counts.pop("<unk>", None)
    return token_counts


def rank_tokens(token_counts: dict) -> list:
    """
    Ranks distinct tokens by decreasing occurrence count.
    Breaks ties using Python string order (ascending/alphabetical).
    """
    return sorted(
        token_counts.keys(),
        key=lambda term: (-token_counts[term], term)
    )


def create_vocab(ranked_tokens: list, max_vocab_size: int) -> dict:
    """
    Builds the vocabulary dictionary:
    - Reserves '<pad>' at ID 0 and '<unk>' at ID 1.
    - Keeps at most (max_vocab_size - 2) regular ranked tokens with consecutive IDs starting from 2.
    """
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }
    
    num_allowed_regular = max(0, max_vocab_size - 2)
    selected_tokens = ranked_tokens[:num_allowed_regular]
    
    for idx, token in enumerate(selected_tokens, start=2):
        vocab[token] = idx
        
    return vocab


def encode_corpus(tokenized_corpus: list, vocab: dict) -> list:
    """
    Encodes each sentence into integer lists using the vocabulary.
    Replaces any unknown or excluded tokens with the '<unk>' token ID (ID 1).
    """
    unk_id = vocab.get("<unk>", 1)
    
    encoded = []
    for tokens in tokenized_corpus:
        encoded_sent = [vocab.get(token, unk_id) for token in tokens]
        encoded.append(encoded_sent)
        
    return encoded


def build_vocabulary(corpus: list, max_vocab_size: int) -> dict:
    """
    Returns a dictionary with vocab (token IDs) and encoded (integer lists).
    """
    tokenized_corpus = tokenize_corpus(corpus)
    token_counts = count_tokens(tokenized_corpus)
    ranked_tokens = rank_tokens(token_counts)
    vocab = create_vocab(ranked_tokens, max_vocab_size)
    encoded = encode_corpus(tokenized_corpus, vocab)
    
    return {
        "vocab": vocab,
        "encoded": encoded
    }