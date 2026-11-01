import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    """
    Returns the state values as a NumPy array.
    
    DRY RUN EXAMPLE:
    Input: 
        episodes = [[[0, 1], [1, 2], [2, 3]]]  (1 episode: [state, reward] steps)
        gamma = 1                               (No discounting)
        n_states = 3                            (States are 0, 1, and 2)
        
    Step 1: Backward Return Calculation (T = 3)
        We move from the end of the episode to the start:
        - Step t = 2 (State 2, Reward 3): G = 3 + (1 * 0) = 3.0
        - Step t = 1 (State 1, Reward 2): G = 2 + (1 * 3) = 5.0
        - Step t = 0 (State 0, Reward 1): G = 1 + (1 * 5) = 6.0
        
    Step 2: First-Visit Filtering
        We check each step forward and record each state's return once per episode:
        - State 0 gets return 6.0 -> returns[0] = [6.0]
        - State 1 gets return 5.0 -> returns[1] = [5.0]
        - State 2 gets return 3.0 -> returns[2] = [3.0]
        
    Step 3: Averaging Returns
        - State 0 average: mean([6.0]) = 6.0
        - State 1 average: mean([5.0]) = 5.0
        - State 2 average: mean([3.0]) = 3.0
        
    Output: 
        array([6., 5., 3.])
    -------------------------------------------------------------------------
    """
    # Create an empty list for each state to store all its collected returns (scores)
    returns = [[] for _ in range(n_states)]
    
    # Go through each complete episode (run/game) one by one
    for episode in episodes:
        if not episode:
            continue
            
        T = len(episode)
        episode_returns = [0.0] * T
        G = 0.0
        
        # Step 1: Calculate the total return (reward) from the end of the episode back to the start.
        # We go backwards because future rewards determine the value of current states!
        for t in range(T - 1, -1, -1):
            reward = episode[t][1]          # Get the reward at step t
            G = reward + gamma * G          # Add current reward plus discounted future return (gamma)
            episode_returns[t] = G
            
        # Step 2: First-Visit Monte Carlo trick!
        # We only keep the FIRST time we visit a state in this episode to keep our learning unbiased.
        visited = set()
        for t in range(T):
            state = episode[t][0]           # Get the state at step t
            if state not in visited:
                visited.add(state)          # Mark this state as seen for this episode
                returns[state].append(episode_returns[t]) # Save its return score
                
    # Step 3: Calculate the average return for each state and format it as a NumPy array
    values = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        if returns[s]:                      # If we actually visited this state at least once
            values[s] = np.mean(returns[s]) # Take the average of all its recorded returns
            
    return np.round(values, 4)