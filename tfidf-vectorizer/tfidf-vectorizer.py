import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    # Step 1 : Normalize each document 
    tokenized_docs = [ ]
    for doc in documents:
        tokenized_doc = doc.lower().split()
        tokenized_docs.append(tokenized_doc)
    # Step 2 : Prepare Vocabulary in alphabetically sorted order from tokenized_docs
    vocabulary = set([ ])
    for doc in tokenized_docs:
        for token in doc: 
            vocabulary.add(token)
    vocabulary = sorted(list(vocabulary))
    # Step 3 : Get index for each token in vocabulary
    index_map = { }
    for index, token in enumerate(vocabulary):
        index_map[token] = index 
    num_documents = len(tokenized_docs)
    vocab_size = len(vocabulary)
    # Step 4 : Compute document frequency
    # df[token] = number of documents containing token
    document_frequency = {}
    for token in vocabulary:
        document_frequency[token] = 0
    for doc in tokenized_docs:
        # We only want to count a word once per document.
        # Example: "cat cat dog"
        # cat appears twice, but its document frequency
        # contribution from this document is still only 1.
        seen_tokens = set()
        for token in doc:
            seen_tokens.add(token)
        for token in seen_tokens:
            document_frequency[token] += 1

    # Step 5: Compute IDF for every vocabulary word
    idf = {}
    for token in vocabulary:
        df = document_frequency[token]
        idf[token] = math.log(num_documents / df)

    
    # Step 6: Allocate TF-IDF matrix
    # rows    -> documents
    # columns -> vocabulary words
    matrix = np.zeros((num_documents, vocab_size), dtype=np.float64)

    # Step 7: Compute TF-IDF for each document
    for doc_index, doc in enumerate(tokenized_docs):
        # Empty document has no TF-IDF values
        if len(doc) == 0:
            continue

        # Count terms manually — no Counter
        term_counts = {}

        for token in doc:
            if token not in term_counts:
                term_counts[token] = 1
            else:
                term_counts[token] += 1

        # Compute TF-IDF
        for token, count in term_counts.items():
            # Term frequency
            tf = count / len(doc)
            # TF-IDF
            tfidf = tf * idf[token]
            # Find the correct matrix column
            column_index = index_map[token]
            matrix[doc_index, column_index] = tfidf
            
    return {
        "vocabulary" : vocabulary,
        "tfidf_matrix" : matrix,
    }