def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    # Total number of users being evaluated
    total_users = len(recommendations)
    
    if total_users > 0:
        users_with_relevant_hits = 0
        
        # Iterate through each user's recommendations and their corresponding ground truth labels
        for user_recommendation, truth in zip(recommendations, ground_truth):
            # Extract the top-k recommendations and convert to a set for O(1) lookup performance
            top_k = set(user_recommendation[0:k])
            
            # Check if any ground truth label exists in the user's top-k recommendations
            for truth_label in truth:
                if truth_label in top_k:
                    users_with_relevant_hits += 1
                    break  # Count each user at most once, then move to the next user
                    
        # Return the fraction of users who had at least one relevant hit in their top-k list
        return users_with_relevant_hits / total_users
        
    return 0.0