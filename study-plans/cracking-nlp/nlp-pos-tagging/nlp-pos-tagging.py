def pos_tagging(train, test, default_tag="NN"):
    """
    Returns: list
    """
    # Step 1 : get dictionary from training set 
    # key : word and value : dictionary of {key : tag, value : count}
    word_to_tag_count_map = {}
    for sentence in train:
        for item in sentence:
            word, label = item 
            if word not in word_to_tag_count_map:
                word_to_tag_count_map[word] = {}
            if label not in word_to_tag_count_map[word]:
                word_to_tag_count_map[word][label] = 0
            word_to_tag_count_map[word][label] += 1
    # Step 2 : Now, prepare model artifact from word_to_tag_count_map 
    # model_artifact is a dictionary, where key is word and value is its tag, based on count 
    model_artifact = {}
    for word in word_to_tag_count_map:
        count_map = word_to_tag_count_map[word]
        # Sort on the basis of decreasing count and lexicographical order of tag
        result = sorted(count_map.items(), key=lambda item: (-item[1], item[0]))
        # Extract the tag
        model_artifact[word] = result[0][0]
    # Step 3 : Do inference on test set from model_artifact
    result = [ ]
    for sentence in test:
        pos_tagged_output = [ ]
        for word in sentence:
            if word not in model_artifact:
                pos_tagged_output.append(default_tag)
            else:
                pos_tagged_output.append(model_artifact[word])
        result.append(pos_tagged_output)
    return result