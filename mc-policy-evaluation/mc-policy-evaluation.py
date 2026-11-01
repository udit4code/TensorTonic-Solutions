import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    """
    Returns the state values as a NumPy array.
    """
    returns = [[] for _ in range(n_states)]
    
    for episode in episodes:
        if not episode:
            continue
            
        T = len(episode)
        episode_returns = [0.0] * T
        G = 0.0
        
        # Step 1 : Compute every episode's return backward
        for t in range(T - 1, -1, -1):
            reward = episode[t][1]
            G = reward + gamma * G
            episode_returns[t] = G
            
        # Step 2 : Use only the earliest occurrence of each state per episode (First-Visit MC)
        visited = set()
        for t in range(T):
            state = episode[t][0]
            if state not in visited:
                visited.add(state)
                returns[state].append(episode_returns[t])
                
    # Step 3 : Average collected returns and format as a NumPy array
    values = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        if returns[s]:
            values[s] = np.mean(returns[s])
            
    return np.round(values, 4)