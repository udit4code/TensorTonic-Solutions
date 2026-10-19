def tokenize(text: str) -> list:
    """
    Returns a list of token strings by scanning character by character.
    """
    tokens = []
    current_word = []
    
    for char in text:
        # Check if the character is an ASCII letter, digit, or underscore
        is_ascii_alnum_underscore = (
            ('a' <= char <= 'z') or 
            ('A' <= char <= 'Z') or 
            ('0' <= char <= '9') or 
            (char == '_')
        )
        
        if is_ascii_alnum_underscore:
            current_word.append(char)
        else:
            # If we were building an alphanumeric/underscore token, flush it first
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            
            # Discard whitespace, emit any other character as its own token
            if not char.isspace():
                tokens.append(char)
                
    # Flush any remaining word at the end of the string
    if current_word:
        tokens.append("".join(current_word))
        
    return tokens