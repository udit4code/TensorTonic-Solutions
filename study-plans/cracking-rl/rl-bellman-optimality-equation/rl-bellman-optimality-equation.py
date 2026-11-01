def bellman_optimality_backup(P: list, R: list, gamma: float, V: list) -> list:
    """
        P and R are nested lists of shape (S, A, S), with S states and A actions. 
        The axes are current state, action and next state, in that order. 
        V is a list of S finite values 
        Returns one updated state value per state, rounded to four decimals.
    """
    
    # Step 1: Initialize dimensions and output list
    # Step 1.1: Determine the number of states (S) and actions (A)
    S = len(V)
    A = len(P[0])
    
    # Step 1.2: Initialize the list to store the new backup values for each state
    V_new = []
    
    # Step 2: Iterate over each state to compute its updated value
    for s in range(S):
        action_values = []
        
        # Step 3: Compute Q-values for each action available in the current state
        for a in range(A):
            q_sa = 0.0
            
            # Step 3.1: Sum over all possible next states using transition probabilities, rewards, and discounted values
            for next_s in range(S):
                q_sa += P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
                
            action_values.append(q_sa)
        
        # Step 4: Apply the Bellman Optimality operator
        # Step 4.1: Find the maximum Q-value across all actions for state s
        # Step 4.2: Round the result to four decimals and store it
        V_new.append(round(max(action_values), 4))
        
    # Step 5: Return the fully updated list of state values
    return V_new