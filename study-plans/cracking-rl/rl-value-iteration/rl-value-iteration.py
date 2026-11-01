def value_iteration(P: list, R: list, gamma: float, tol: float = 1e-6, max_iters: int = 1000) -> tuple:
    """
    Returns a Python tuple (V, policy): a list of state-value floats rounded to four decimals, 
    followed by a list of integer action indices, one entry per state.
    Uses final values to select a greedy action, choosing the lowest action index on ties.
    """
    S = len(P)
    A = len(P[0])
    
    # Step 1: Initialize the value function with zeros for all states
    V = [0.0] * S
    
    # Step 2: Iterate until convergence or max iterations are reached
    for iteration in range(max_iters):
        delta = 0.0
        V_new = [0.0] * S
        
        # Step 3: Iterate over every state to compute updated values
        for s in range(S):
            action_values = []
            
            # Step 4: Compute Q-values for each available action in state s
            for a in range(A):
                q_sa = 0.0
                for next_s in range(S):
                    q_sa += P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
                action_values.append(q_sa)
            
            # Step 5: Apply the Bellman Optimality operator (store max Q-value)
            V_new[s] = max(action_values)
            
            # Step 6: Track the maximum change across all states for convergence checking
            delta = max(delta, abs(V[s] - V_new[s]))
        
        # Update the value function for the next iteration
        V = V_new
        
        # Step 7: Check if the values have converged within the given tolerance
        if delta < tol:
            break
            
    # Step 8: Use the final converged values to extract the greedy policy
    policy = [0] * S
    for s in range(S):
        action_values = []
        
        # Step 8.1: Compute final Q-values for state s using the converged V
        for a in range(A):
            q_sa = 0.0
            for next_s in range(S):
                q_sa += P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
            action_values.append(q_sa)
        
        # Step 8.2: Select the greedy action with tie-breaking (lowest action index on ties)
        # Using strict '>' ensures that if a later action equals max_q, it is ignored 
        # in favor of the earlier action index (lowest index).
        best_action = 0
        max_q = action_values[0]
        for a in range(1, A):
            if action_values[a] > max_q:
                max_q = action_values[a]
                best_action = a
                
        policy[s] = best_action
        
    # Step 9: Round each state value in V to four decimal places
    rounded_values = [round(v, 4) for v in V]
    
    # Step 10: Return the required tuple (V, policy)
    return (rounded_values, policy)