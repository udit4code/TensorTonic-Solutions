def named_entity_recognition(sentences, entity_dict):
    """
    Returns: list
    """
    result = [ ]
    otherwise_tag = "O"
    for sentence in sentences:
        ner_tagged_output = [ ]
        for word in sentence:
            if word not in entity_dict:
                ner_tagged_output.append(otherwise_tag)
            else:
                ner_tagged_output.append(entity_dict[word])
        result.append(ner_tagged_output)
    return result 