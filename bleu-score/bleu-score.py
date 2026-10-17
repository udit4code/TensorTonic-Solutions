import math
from collections import defaultdict 

def get_n_grams(tokens: list, n: int) -> defaultdict: 
    """
        Returns a defaultdict object of all n-grams of size n from the token list.
        Example: tokens=['the', 'cat', 'sat'], n=2 -> {('the', 'cat'): 1, ('cat', 'sat'): 1}
    """
    freq_map = defaultdict(int)
    for start in range(0, len(tokens) - n + 1, 1):
        end = start + n 
        n_gram_candidate = tuple(tokens[start : end])
        freq_map[n_gram_candidate] += 1
    return freq_map
    
def get_modified_precision(candidate: list, reference: list, n: int) -> float:
    """
        Calculates clipped n-gram precision for a specific n.
    """
    cand_n_grams = get_n_grams(candidate, n)
    ref_n_grams = get_n_grams(reference, n)
    print(f"For n = {n}, cand_n_grams : {cand_n_grams}\n and ref_n_grams : {ref_n_grams}\n")
    
    # Clip candidate counts by reference counts
    clipped_intersection = 0 
    for n_gram, candidate_count in cand_n_grams.items():
        clipped_intersection += min(candidate_count, ref_n_grams[n_gram])

    total_cand_n_grams = sum(cand_n_grams.values())
    
    if total_cand_n_grams == 0:
        return 0.0
        
    return clipped_intersection / total_cand_n_grams


def get_brevity_penalty(candidate: list, reference: list) -> float:
    """
    Calculates the brevity penalty (BP) factor.
    """
    c = len(candidate)
    r = len(reference)
    
    if c > r:
        return 1.0
    elif c == 0:
        return 0.0
    else:
        return math.exp(1.0 - r / c)
        
def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """

    # Step 1 : Compute Brevity Penalty
    bp = get_brevity_penalty(candidate, reference)
    
    # Step 2 : Compute modified precision for each n from 1 to max_n
    precisions = [ ]
    for n in range(1, max_n + 1, 1):
        p_n = get_modified_precision(candidate, reference, n)
        if p_n == 0.0:
            # Unsmoothed BLEU drops instantly to 0 if any precision is 0
            return 0.0 
        precisions.append(math.log(p_n))
    # Step 3 : Compute geometric mean of precisions using uniform weights (1 / max_n)
    geometric_mean = math.exp(sum(precisions) / max_n)
    
    # Step 4 : Multiply by brevity penalty and return
    return bp * geometric_mean