import numpy as np

def compute_perplexity(token_ids: list, logits: list) -> dict:
    """
    Computes token probabilities, NLLs, average NLL, perplexity, and sequence log-probability.
    
    Args:
        token_ids: List of integer token IDs.
        logits: List or array of shape (seq_len, vocab_size) representing model output logits.
        
    Returns:
        dict: A dictionary containing:
            - token_probabilities: list of floats
            - token_nlls: list of floats
            - avg_nll: float
            - perplexity: float
            - sequence_log_prob: float
    """
    # Convert into numpy arrays
    token_ids = np.array(token_ids)
    logits = np.array(logits)
    
    if token_ids.ndim > 1:
        token_ids = token_ids.squeeze()
        
    # Robust alignment for causal language modeling: 
    # WHY WE DO THIS:
    # Language models work by predicting the next token. 
    # If our text has 3 tokens -> [The, cat, sat], the model outputs predictions 
    # for what comes after "The" and what comes after "cat". 
    # We have to line up these predictions (logits) with the actual target tokens 
    # they are trying to guess.

    # Case 1: Logits are already pre-shifted (shorter by 1).
    # Example: 
    #   token_ids = [10, 20, 30] (length 3)
    #   logits    = [pred_for_20, pred_for_30] (length 2)
    # 
    # Here, logits is already cut down to match predictions. We just drop the 
    # very first token ([10]) so token_ids becomes [20, 30] and matches length 2.
    if len(token_ids) == len(logits) + 1 and len(token_ids) > 1:
        token_ids = token_ids[1:]
    # Case 2: Logits and token_ids have the exact same full length (unshifted).
    # Example: 
    #   token_ids = [10, 20, 30] (length 3)
    #   logits    = [pred_for_10, pred_for_20, pred_for_30] (length 3)
    # 
    # Here, the last logit tries to predict a token *after* 30, which we don't have.
    # And the first logit tries to predict 20. So we fix both:
    # 1. logits[:-1] drops the last prediction -> leaves [pred_for_10, pred_for_20]
    # 2. token_ids[1:] drops the first token [10] -> leaves [20, 30]
    elif len(logits) == len(token_ids) and len(token_ids) > 1:
        logits = logits[:-1]
        token_ids = token_ids[1:]
    
    # Numerically stable softmax: subtract max logit along vocabulary dimension
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Extract the probability assigned to the actual target tokens
    # HOW NUMPY FANCY INDEXING WORKS HERE:
    # We want to pull out a specific probability from every row of our 'probs' matrix.
    # 'probs' has shape (seq_len, vocab_size) -> rows are positions, columns are vocabulary tokens.
    # 1. np.arange(len(token_ids)) creates row indices: [0, 1, 2, ...] (each position in the text)
    # 2. token_ids gives the column indices (the exact correct token IDs we want to check)
    # By passing both lists into NumPy like probs[[row_indices], [column_indices]], 
    # it pairs them up to grab exactly the probability of the right answer at every step.
    token_probabilities = probs[np.arange(len(token_ids)), token_ids]
    
    # Clip probabilities to prevent log(0) errors
    eps = 1e-12
    token_probabilities_clipped = np.clip(token_probabilities, eps, 1.0)
    
    # Compute metrics
    token_nlls = -np.log(token_probabilities_clipped)
    avg_nll = float(np.mean(token_nlls))
    perplexity = float(np.exp(avg_nll))
    sequence_log_prob = float(np.sum(np.log(token_probabilities_clipped)))
    
    return {
        "token_probabilities": token_probabilities.tolist(),
        "token_nlls": token_nlls.tolist(),
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "sequence_log_prob": sequence_log_prob
    }