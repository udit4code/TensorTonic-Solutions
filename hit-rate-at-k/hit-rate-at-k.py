def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    total_users = len(recommendations)
    if total_users > 0:
        users_with_relevant_hits = 0
        for user_recommendation, truth in zip(recommendations, ground_truth):
            top_k = set(user_recommendation[0:k])
            for truth_label in truth:
                if truth_label in top_k:
                    users_with_relevant_hits += 1
                    break
        return (users_with_relevant_hits / total_users)
    return 0.0