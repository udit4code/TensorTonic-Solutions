def bellman_expectation_backup(P: list, R: list, policy: list, gamma: float, V: list) -> list:
    """
    Returns one updated state value per state, rounded to four decimals.
    """
    # Find out how many total states we have in our environment
    num_states = len(P)
    new_V = []  # This list will store our newly calculated value for each state
    
    # Step 1: Loop through every single state (s)
    for s in range(num_states):
        v_s = 0.0  # Initialize the total expected value for state s
        num_actions = len(P[s])  # Number of possible actions from state s
        
        # Step 2: Loop through every possible action (a) we can take from state s
        for a in range(num_actions):
            pi_a = policy[s][a]  # Probability of choosing action 'a' in state 's' under our policy
            
            # If our policy never takes this action, we can skip it to save computer power
            if pi_a == 0.0:
                continue
            
            action_value = 0.0  # Expected value for taking this specific action
            
            # Step 3: Loop through all possible next states (s_prime) we could end up in
            for s_prime in range(len(P[s][a])):
                p_transition = P[s][a][s_prime]  # Probability of reaching s_prime from s via action a
                
                # If this transition is impossible, skip it
                if p_transition == 0.0:
                    continue
                
                # Handle different reward formats: 
                # Checks if rewards depend on the next state R[s][a][s_prime] or just action R[s][a]
                # reward is a triplet indexed by (s, a, s_prime)
                reward = R[s][a][s_prime] if isinstance(R[s][a], list) else R[s][a]
                
                # Add to our action value:
                # Transition Probability * (Immediate Reward + Discounted Value of the Next State)
                action_value += p_transition * (reward + gamma * V[s_prime])
                
            # Weight the action value by how likely we are to choose that action, then add it to the state's total value
            v_s += pi_a * action_value
            
        # Round the final calculated value for state s to 4 decimal places and save it
        new_V.append(round(v_s, 4))
        
    return new_V