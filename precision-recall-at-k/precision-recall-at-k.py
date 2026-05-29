def get_intersection(recommended, relevant, k):
    return set(recommended[0:k]).intersection(set(relevant))
    
def get_precission_at_k(recommended, relevant, k):
    intersection = get_intersection(recommended, relevant, k)
    return len(intersection) / k

def get_recall_at_k(recommended, relevant, k):
    intersection = get_intersection(recommended, relevant, k)
    return len(intersection) / len(set(relevant))
    
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    precission = get_precission_at_k(recommended, relevant, k)
    recall = get_recall_at_k(recommended, relevant, k)
    return [precission, recall]