import heapq as hq 

def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    rated = set(rated_indices)
    heap = []   # stores (score, index)
    for idx, score in enumerate(scores):
        # skip already rated items
        if idx not in rated:
            if len(heap) < k:
                hq.heappush(heap, (score, -idx))
            else:
                # current score beats smallest in heap
                if score > heap[0][0]:
                    hq.heapreplace(heap,(score, -idx))

    # heap contains k best items,
    # but smallest is first
    result = sorted(heap,reverse=True)
    return [-idx for score, idx in result]
    