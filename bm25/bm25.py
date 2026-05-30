import numpy as np
import math
from collections import defaultdict

def get_frequency_maps(docs):
    document_frequency_map = defaultdict(int)
    term_frequency_map = {}
    doc_lengths = {}

    for doc_id, doc in enumerate(docs):
        doc_lengths[doc_id] = len(doc)
        term_frequency_map[doc_id] = defaultdict(int)

        for word in doc:
            term_frequency_map[doc_id][word] += 1

        for word in term_frequency_map[doc_id]:
            document_frequency_map[word] += 1

    return (
        document_frequency_map,
        term_frequency_map,
        doc_lengths
    )
    
def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    document_frequency_map, term_frequency_map, doc_lengths = get_frequency_maps(docs)
    N = len(docs)

    avg_doc_length = (sum(doc_lengths.values()) / N if N > 0 else 0)

    scores = np.zeros(N, dtype=float)

    for doc_id in range(N):
        doc_length = doc_lengths[doc_id]

        score = 0.0
        for term in query_tokens:
            df = document_frequency_map.get(term, 0)
            if df != 0:
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf = term_frequency_map[doc_id].get(term, 0)
    
                if tf != 0:
                    numerator = tf * (k1 + 1)
                    denominator = (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    score += idf * (numerator / denominator)

        scores[doc_id] = score

    return scores