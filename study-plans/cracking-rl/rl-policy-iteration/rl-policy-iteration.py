def policy_iteration(P: list, R: list, gamma: float, eval_tol: float = 1e-8, max_iters: int = 200) -> tuple:
    """
    Returns (values, policy): rounded float values and integer actions, one per state.
    Implemented using naive for loops.
    """
    S = len(P)
    A = len(P[0])
    
    # Step 1: Initialize policy arbitrarily (action 0 for all states) and value function with zeros
    policy = [0] * S
    V = [0.0] * S
    
    # Step 2: Main policy iteration loop
    for iteration in range(max_iters):
        
        # Step 3: Policy Evaluation Phase
        # Iteratively compute the value function for the current fixed policy until convergence
        while True:
            delta = 0.0
            V_new = [0.0] * S
            
            for s in range(S):
                a = policy[s]
                q_sa = 0.0
                
                # Compute expected value following the current policy action 'a'
                for next_s in range(S):
                    q_sa += P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
                    
                V_new[s] = q_sa
                delta = max(delta, abs(V[s] - V_new[s]))
                
            V = V_new
            
            # Check if value evaluation has converged within the tolerance
            if delta < eval_tol:
                break
        
        # Step 4: Policy Improvement Phase
        policy_stable = True
        new_policy = list(policy)
        
        for s in range(S):
            action_values = []
            
            # Step 4.1: Compute Q-values for all available actions in state s
            for a in range(A):
                q_sa = 0.0
                for next_s in range(S):
                    q_sa += P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
                action_values.append(q_sa)
            
            # Step 4.2: Select the greedy action with tie-breaking (lowest action index on ties)
            # Using strict '>' ensures earlier action indices are preferred on equal Q-values.
            best_action = 0
            max_q = action_values[0]
            for a in range(1, A):
                if action_values[a] > max_q:
                    max_q = action_values[a]
                    best_action = a
            
            # Step 4.3: Check if policy changed for this state
            if best_action != policy[s]:
                policy_stable = False
                
            new_policy[s] = best_action
            
        policy = new_policy
        
        # Step 5: Terminate if the policy has stabilized (no changes made)
        if policy_stable:
            break
            
    # Step 6: Round final state values to four decimal places and return the required tuple
    rounded_values = [round(v, 4) for v in V]
    
    return (rounded_values, policy)