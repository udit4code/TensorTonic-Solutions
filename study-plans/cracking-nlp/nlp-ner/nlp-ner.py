def named_entity_recognition(sentences, entity_dict):
    """
    Args:
        sentences: List[List[str]]
        entity_dict: Dict[str, str]

    Returns:
        List[List[str]]
    """
    otherwise_tag = "O"

    def get_ner_tagged_sentence(sentence):
        ner_tagged_output = []
        for word in sentence:
            if word in entity_dict:
                ner_tagged_output.append(entity_dict[word])
            else:
                ner_tagged_output.append(otherwise_tag)
        return ner_tagged_output

    return list(map(get_ner_tagged_sentence, sentences))