def rgb_to_grayscale(image: list) -> list:
    """
    Returns the H-by-W grayscale list of floats, rounded to four decimals.
    """
    transform = lambda r, g, b : 0.299*r + 0.587*g + 0.114*b 
    return [[round(transform(r, g, b), 4) for r, g, b in row] for row in image]