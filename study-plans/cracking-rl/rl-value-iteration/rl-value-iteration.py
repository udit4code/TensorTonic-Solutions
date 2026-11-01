import numpy as np

def value_iteration(P: list, R: list, gamma: float, tol: float = 1e-6, max_iters: int = 1000) -> tuple:
    """
    Returns a Python tuple (V, policy): a list of state-value floats rounded to four decimals, 
    followed by a list of integer action indices, one entry per state, optimized with NumPy.
    """
    # Step 1: Convert input lists to NumPy arrays for vectorized operations
    P_arr = np.array(P)
    R_arr = np.array(R)
    S = P_arr.shape[0]
    
    # Step 2: Initialize the value function array with zeros for all states
    V = np.zeros(S)
    
    # Step 3: Iterate until convergence or max iterations are reached
    for iteration in range(max_iters):
        # Step 3.1: Compute Q-values for all state-action pairs simultaneously (shape: S x A)
        # Broadcasting handles gamma * V across the next-state dimension, 
        # and np.sum reduces over the next-state axis (-1).
        q_values = np.sum(P_arr * (R_arr + gamma * V), axis=-1)
        
        # Step 3.2: Apply the Bellman Optimality operator by taking the max over actions (axis=1)
        V_new = np.max(q_values, axis=1)
        
        # Step 3.3: Compute the maximum change (delta) across all states for convergence testing
        delta = np.max(np.abs(V - V_new))
        
        # Step 3.4: Update value function for the next iteration
        V = V_new
        
        # Step 3.5: Check convergence condition
        if delta < tol:
            break
            
    # Step 4: Compute final Q-values using the converged value function
    q_values_final = np.sum(P_arr * (R_arr + gamma * V), axis=-1)
    
    # Step 5: Extract the greedy policy
    # np.argmax returns the first occurrence of the maximum value along axis=1, 
    # which automatically chooses the lowest action index in the event of ties.
    policy = np.argmax(q_values_final, axis=1).tolist()
    
    # Step 6: Round state values to four decimal places and convert to a standard Python list
    rounded_values = np.round(V, 4).tolist()
    
    # Step 7: Return the required tuple (values, policy)
    return (rounded_values, policy)
    

def value_iteration_naive(P: list, R: list, gamma: float, tol: float = 1e-6, max_iters: int = 1000) -> tuple:
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