import numpy as np 

def bellman_optimality_backup(P: list, R: list, gamma: float, V: list) -> list:
    """
    Returns one updated state value per state, rounded to four decimals,
    optimized using NumPy vectorization.
    """
    # Step 1: Convert input lists to NumPy arrays to enable vectorized operations
    # Step 1.1: P_arr shape is (S, A, S) -> Transition probabilities P(s' | s, a)
    P_arr = np.array(P)
    # Step 1.2: R_arr shape is (S, A, S) -> Rewards R(s, a, s')
    R_arr = np.array(R)
    # Step 1.3: V_arr shape is (S,) -> Current state values V(s')
    V_arr = np.array(V)
    
    # Step 2: Compute Q-values for all state-action pairs simultaneously (shape: S x A)
    # Approach:
    # - gamma * V_arr applies the discount factor to all future state values.
    # - Broadcasting automatically aligns V_arr with the final next-state (s') dimension of R_arr and P_arr.
    # - R_arr + gamma * V_arr computes the bracketed term: R(s, a, s') + gamma * V(s')
    # - Multiplying by P_arr and summing over axis=-1 computes the expected value over all next states s' for each (s, a) pair.
    q_values = np.sum(P_arr * (R_arr + gamma * V_arr), axis=-1)
    
    # Step 3: Apply the Bellman Optimality operator
    # Approach: Find the maximum expected Q-value across all available actions (axis=1) for each state.
    V_new = np.max(q_values, axis=1)
    
    # Step 4: Format and return the output
    # Step 4.1: Round the resulting values to four decimal places.
    # Step 4.2: Convert the NumPy array back to a standard Python list as specified by the type annotation.
    return np.round(V_new, 4).tolist()
    
def bellman_optimality_backup_naive(P: list, R: list, gamma: float, V: list) -> list:
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