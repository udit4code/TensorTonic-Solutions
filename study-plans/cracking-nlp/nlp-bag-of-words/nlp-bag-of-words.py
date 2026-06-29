
def bag_of_words(corpus):
    """
    Returns: dict
    """
    # Step 1 : Get words from the corpus into a set 
    vocabulary = set({})
    for document in corpus: 
        for word in document:
            vocabulary.add(word)
    # Step 2 : Organize the words into a sorted order
    vocabulary = sorted(list(vocabulary))
    # Step 3 : Get word_to_index_map from vocabulary
    word_to_index_map = {}
    for index, word in enumerate(vocabulary):
        word_to_index_map[word] = index
    # Size of Vocabulary
    V = len(word_to_index_map)
    # Step 4 : Form embedding for each document in corpus 
    embeddings = [ ]
    for document in corpus: 
        document_embedding = [0] * V 
        for word in document:
            index = word_to_index_map[word]
            document_embedding[index] += 1
        embeddings.append(document_embedding)
    return {
        "vocab" : word_to_index_map,
        "vectors" : embeddings,
    }