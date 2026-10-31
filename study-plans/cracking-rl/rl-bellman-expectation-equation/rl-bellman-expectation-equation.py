
import numpy as np

# Instead of using nested for loops over states, actions, and next states, 
# this version uses NumPy broadcasting and tensor dot-products (np.sum). 
# It runs entirely in C, making it lightning-fast for large Markov Decision Processes.
def bellman_expectation_backup(P: list, R: list, policy: list, gamma: float, V: list) -> np.ndarray:
    """
    Performs a vectorized Bellman expectation backup using NumPy, rounded to four decimals.
    
    Expected Shapes:
    - P: (num_states, num_actions, num_states)
    - R: (num_states, num_actions) or (num_states, num_actions, num_states)
    - policy: (num_states, num_actions)
    - V: (num_states,)
    """
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    policy = np.asarray(policy, dtype=float)
    V = np.asarray(V, dtype=float)
    
    # If R is 2D shape (S, A), expand it to (S, A, 1) for broadcasting against P and V
    if R.ndim == 2:
        R = R[..., None]
        
    # 1. Compute Q-values for all state-action pairs:
    # Q(s, a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
    # V automatically broadcasts across the last axis (s') -> shape (1, 1, S')
    Q = np.sum(P * (R + gamma * V), axis=-1)  # Result shape: (num_states, num_actions)
    
    # 2. Compute expected state values under the policy:
    # V_new(s) = sum_a policy(a|s) * Q(s, a)
    V_new = np.sum(policy * Q, axis=1)        # Result shape: (num_states,)

    return np.round(V_new, 4).tolist()
    

# This is the Naive Implementation, using Nested For-Loops. 
def bellman_expectation_backup_naive(P: list, R: list, policy: list, gamma: float, V: list) -> list:
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