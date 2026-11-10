import re

def text_normalize(text: str, operations: list) -> str:
    """
    Returns the normalized string.
    """
    for op in operations:
        if op == 'lowercase':
            text = text.lower()
        elif op == 'remove_punctuation':
            text = "".join(ch for ch in text if (ch.isalnum() or ch.isspace()) and ch != '_')
        elif op == 'remove_digits':
            text = re.sub(r'\d', '', text)
        elif op == 'collapse_whitespace':
            text = re.sub(r'\s+', ' ', text)
        elif op == 'strip':
            text = text.strip()
    return text