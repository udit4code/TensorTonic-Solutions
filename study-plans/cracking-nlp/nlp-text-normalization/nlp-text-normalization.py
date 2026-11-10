import re

def text_normalize(text: str, operations: list) -> str:
    """
    Normalizes input text by sequentially applying a list of specified operations.
    
    Supported operations:
    - 'lowercase': Converts all characters to lowercase.
    - 'remove_punctuation': Removes non-alphanumeric, non-whitespace characters (including underscores).
    - 'remove_digits': Removes all decimal digits (0-9).
    - 'collapse_whitespace': Replaces consecutive whitespace runs with a single ordinary space.
    - 'strip': Removes leading and trailing whitespace.
    
    Args:
        text (str): The input string to normalize.
        operations (list): A list of operation names to apply in order.
        
    Returns:
        str: The fully normalized text string.
    """
    # Iterate through each requested operation in the exact order provided
    for op in operations:
        if op == 'lowercase':
            # Convert string to lowercase using Python's Unicode-aware lower() method
            text = text.lower()
            
        elif op == 'remove_punctuation':
            # Keep only alphanumeric characters and whitespaces, explicitly filtering out underscores
            text = "".join(ch for ch in text if (ch.isalnum() or ch.isspace()) and ch != '_')
            
        elif op == 'remove_digits':
            # Remove all decimal digits using regular expressions (\d matches any decimal digit)
            text = re.sub(r'\d', '', text)
            
        elif op == 'collapse_whitespace':
            # Replace any consecutive run of whitespace characters with a single regular space
            text = re.sub(r'\s+', ' ', text)
            
        elif op == 'strip':
            # Remove leading and trailing whitespace from the string
            text = text.strip()
            
    return text