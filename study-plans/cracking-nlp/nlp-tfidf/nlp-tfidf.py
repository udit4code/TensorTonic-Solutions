import math
from collections import defaultdict

def get_tf_map(corpus: list) -> dict:
    # We will use raw term counts. Tokens are case-sensitive.
    # We can think of corpus as list of documents, where each document is a list of words (strings) 
    # We use nested defaultdict to automatically initialize dicts and integers
    tf_map = defaultdict(lambda: defaultdict(int))
    # Iterate through each document in the corpus 
    for doc_id, doc in enumerate(corpus):
        # for each term in the doc, increment the counter
        for term in doc:
            tf_map[term][doc_id] += 1         
    # Convert back to standard dicts
    return {term: dict(docs) for term, docs in tf_map.items()}
    
def get_df_map(tf_map: dict) -> dict:
    # df_map[t] is the number of documents containing the term t
    df_map = {}
    for term, docs in tf_map.items():
        df_map[term] = len(docs)
    return df_map
        
def get_idf_map(tf_map: dict, N: int) -> dict: 
    # We will stick to unsmoothed inverse document frequency
    df_map = get_df_map(tf_map)
    idf_map = {}
    for term, df_val in df_map.items():
        idf_map[term] = math.log(N / df_val)
        # SMOOTHING VERSION : idf_map[term] = math.log((N + 1) / (df_val + 1)) + 1.0
    return idf_map
    
def tfidf(corpus: list) -> dict:
    """
    Returns a dictionary with vocab (token indices) and tfidf (float lists).
    """
    N = len(corpus)
    tf_map = get_tf_map(corpus)
    idf_map = get_idf_map(tf_map, N) 
    
    # Vocabulary sorted alphabetically by Python string order
    vocab = {term: idx for idx, term in enumerate(sorted(tf_map.keys()))}
    
    # Construct TF-IDF vector (list of floats) for each document
    tfidf_list = []
    for doc_id, doc in enumerate(corpus):
        doc_vector = [0.0] * len(vocab)
        for term, idx in vocab.items():
            # Get raw term frequency for this doc, default to 0 if not present
            tf = tf_map.get(term, {}).get(doc_id, 0)
            if tf > 0:
                tf_idf_val = tf * idf_map.get(term, 0.0)
                doc_vector[idx] = round(tf_idf_val, 4)
        tfidf_list.append(doc_vector)
        
    return {
        "vocab": vocab,
        "tfidf": tfidf_list,
    }