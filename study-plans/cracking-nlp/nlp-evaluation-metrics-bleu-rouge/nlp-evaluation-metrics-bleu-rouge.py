import math
from collections import Counter


def get_longest_common_subsequence_length_v1(candidate: list, reference: list) -> int:
    cache = {}
    def lcs_helper(idx, jdx):
        nonlocal cache, candidate, reference
        # Base case : When prefix length of either sequence reaches 0, then, common subsequence does not exist.
        if idx == 0 or jdx == 0:
            return 0 
        if (idx, jdx) not in cache:
            if candidate[idx - 1] == reference[jdx - 1]:
                result = 1 + lcs_helper(idx - 1, jdx - 1)
            else:
                option_1 = lcs_helper(idx - 1, jdx)
                option_2 = lcs_helper(idx, jdx - 1)
                result = max(option_1, option_2)
            cache[(idx, jdx)] = result
        return cache[(idx, jdx)]
        
    return lcs_helper(len(candidate), len(reference))


def get_longest_common_subsequence_length(candidate: list, reference: list) -> int:
    """
    Computes the length of the Longest Common Subsequence (LCS) 
    between candidate and reference using bottom-up Dynamic Programming.
    """
    m, n = len(candidate), len(reference)
    
    # Initialize a 2D DP table of size (m + 1) x (n + 1) filled with zeros.
    # dp[i][j] stores the LCS length of candidate[:i] and reference[:j].
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate[i - 1] == reference[j - 1]:
                # Character match: take diagonal value and add 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Character mismatch: take the maximum of skipping from either sequence
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]


def compute_rouge_l_single(candidate: list, reference: list) -> dict:
    """
    Computes ROUGE-L precision, recall, and F1 between a single candidate 
    and reference using the standard harmonic mean formulation.
    """
    lcs = get_longest_common_subsequence_length(candidate, reference)
    cand_len = len(candidate)
    ref_len = len(reference)
    
    p = lcs / cand_len if cand_len > 0 else 0.0
    r = lcs / ref_len if ref_len > 0 else 0.0
    
    total = cand_len + ref_len
    f1 = (2 * lcs / total) if total > 0 else 0.0
    
    return {"precision": p, "recall": r, "f1": f1}


def compute_rouge_l(candidate: list, references: list) -> dict:
    """
    Computes ROUGE-L against multiple references independently, 
    selecting the reference scores that maximize the F1 score.
    """
    if not candidate or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    best_scores = max(
        (compute_rouge_l_single(candidate, ref) for ref in references),
        key=lambda x: x["f1"]
    )
    
    return {
        "precision": round(best_scores["precision"], 4),
        "recall": round(best_scores["recall"], 4),
        "f1": round(best_scores["f1"], 4),
    }


def get_ngrams(tokens: list, n: int) -> list:
    """
    Extracts a list of n-gram tuples of size n from a token list.
    """
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_rouge_n_single(candidate: list, reference: list, n: int) -> dict:
    """
    Computes ROUGE-N precision, recall, and F1 between a single candidate 
    and reference using frequency counters and clipped overlap.
    """
    cand_ng = Counter(get_ngrams(candidate, n))
    ref_ng = Counter(get_ngrams(reference, n))
    
    matched = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
    cand_total = sum(cand_ng.values())
    ref_total = sum(ref_ng.values())
    
    p = matched / cand_total if cand_total > 0 else 0.0
    r = matched / ref_total if ref_total > 0 else 0.0
    
    total = cand_total + ref_total
    f1 = (2 * matched / total) if total > 0 else 0.0
    
    return {"precision": p, "recall": r, "f1": f1}


def compute_rouge_n(candidate: list, references: list, n: int) -> dict:
    """
    Computes ROUGE-N against multiple references independently, 
    selecting the reference scores that maximize the F1 score.
    """
    if not candidate or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    best_scores = max(
        (compute_rouge_n_single(candidate, ref, n) for ref in references),
        key=lambda x: x["f1"]
    )
    
    return {
        "precision": round(best_scores["precision"], 4),
        "recall": round(best_scores["recall"], 4),
        "f1": round(best_scores["f1"], 4),
    }


def get_closest_reference_length(candidate_len: int, references: list) -> int:
    """
    Finds the reference length closest to the candidate length.
    Ties are broken by preferring the shorter reference length.
    """
    ref_lens = [len(ref) for ref in references]
    return min(ref_lens, key=lambda r: (abs(r - candidate_len), r))


def get_brevity_penalty(candidate_len: int, closest_ref_len: int) -> float:
    """
    Calculates the Brevity Penalty (BP) factor for BLEU.
    """
    if candidate_len >= closest_ref_len:
        return 1.0
    return math.exp(1 - closest_ref_len / candidate_len)


def get_modified_precision(candidate: list, references: list, n: int) -> float:
    """
    Calculates clipped n-gram modified precision across multiple references.
    """
    cand_ngrams = get_ngrams(candidate, n)
    if not cand_ngrams:
        return 0.0
    cand_counts = Counter(cand_ngrams)

    max_ref_counts = Counter()
    for ref in references:
        ref_counts = Counter(get_ngrams(ref, n))
        for ng in ref_counts:
            max_ref_counts[ng] = max(max_ref_counts[ng], ref_counts[ng])

    clipped = sum(min(cand_counts[ng], max_ref_counts.get(ng, 0)) for ng in cand_counts)
    total = sum(cand_counts.values())
    return clipped / total if total > 0 else 0.0


def compute_bleu_score(candidate: list, references: list, max_n: int = 4) -> dict:
    """
    Computes the unsmoothed BLEU score along with precisions and brevity penalty.
    """
    cand_len = len(candidate)
    closest_ref_len = get_closest_reference_length(cand_len, references)
    bp = get_brevity_penalty(cand_len, closest_ref_len)

    precisions = []
    for n in range(1, max_n + 1):
        p_n = get_modified_precision(candidate, references, n)
        precisions.append(p_n)

    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions) / max_n
        bleu = bp * math.exp(log_avg)

    return {
        "bleu": round(bleu, 4),
        "brevity_penalty": round(bp, 4),
        "precisions": [round(p, 4) for p in precisions],
        "closest_ref_length": closest_ref_len
    }


def eval_metrics(candidate: list, references: list, max_n: int = 4) -> dict:
    """
    Returns a unified dictionary of BLEU, precisions, lengths, and ROUGE metrics.
    """
    bleu_dict = compute_bleu_score(candidate, references, max_n)
    rouge_1_dict = compute_rouge_n(candidate, references, 1)
    rouge_2_dict = compute_rouge_n(candidate, references, 2)
    rouge_l_dict = compute_rouge_l(candidate, references)

    return {
        "bleu": bleu_dict["bleu"],
        "brevity_penalty": bleu_dict["brevity_penalty"],
        "precisions": bleu_dict["precisions"],
        "candidate_length": len(candidate),
        "closest_ref_length": bleu_dict["closest_ref_length"],
        "rouge_1": rouge_1_dict,
        "rouge_2": rouge_2_dict,
        "rouge_l": rouge_l_dict,
    }