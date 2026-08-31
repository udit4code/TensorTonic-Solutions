import math
import numpy as np 

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    N = len(actual_tokens)
    log_sum = 0.0
    
    for i in range(N):
        p = prob_distributions[i][actual_tokens[i]]
        log_sum += math.log(p)
    H = -log_sum / N
    
    return round(math.exp(H), 4)