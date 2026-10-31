def discounted_returns(rewards: list, gamma: float) -> list:
    """
    Returns one discounted return per reward, rounded to four decimals.
    """
    returns = [0.0] * len(rewards)
    G = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + gamma * G
        returns[i] = round(G, 4)
    return returns