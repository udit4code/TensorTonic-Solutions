import math
import torch

def causal_attention(Q, K, V):
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    seq_q = scores.shape[-2]
    seq_k = scores.shape[-1]

    # Boolean mask.
    #
    # True means:
    # "Mask this position."
    mask = torch.triu(
        torch.ones(
            seq_q,
            seq_k,
            dtype=torch.bool,
            device=Q.device
        ),
        diagonal=1
    )

    # Replace future-token attention scores with -inf.
    scores = scores.masked_fill(mask, float("-inf"))

    attention = torch.softmax(scores, dim=-1)

    return attention @ V