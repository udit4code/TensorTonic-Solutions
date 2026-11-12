def segment_sentences(text: str) -> list:
    """
    Returns a list of sentence strings.
    """
    if not text:
        return []
    
    abbreviations = {"Mr", "Mrs", "Dr", "Jr", "Sr", "vs", "etc", "i.e", "e.g", "U.S"}
    sentences = []
    start = 0
    i = 0
    n = len(text)
    
    while i < n:
        c = text[i]
        if c in ('.', '!', '?'):
            is_abbrev = False
            if c == '.':
                j = i - 1
                while j >= start and not text[j].isspace():
                    j -= 1
                token = text[j+1:i]
                if token in abbreviations:
                    is_abbrev = True
            
            if not is_abbrev:
                is_boundary = False
                if i == n - 1:
                    is_boundary = True
                else:
                    rem = text[i+1:]
                    if rem and rem[0].isspace():
                        next_non_ws = None
                        for ch in rem:
                            if not ch.isspace():
                                next_non_ws = ch
                                break
                        if next_non_ws is None or next_non_ws.isupper():
                            is_boundary = True
                
                if is_boundary:
                    sentence = text[start:i+1].strip()
                    if sentence:
                        sentences.append(sentence)
                    start = i + 1
        i += 1
        
    if start < n:
        trailing = text[start:].strip()
        if trailing:
            sentences.append(trailing)
            
    return sentences