import numpy as np

def kmeans(X, k, max_iters=100, seed=42):
    """
    Returns: tuple of (labels as list[int], centroids as list[list[float]])
    """
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    # Step 1 : Initialize centroids by randomly selecting  k distinct data points 
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices].copy() # Why copy ? Because X[indices]
    # Step 2 : Repeat until convergence
    for iteration_id in range(max_iters):
        # Step 2.1 : Assign each point to its nearest centroid using Euclidean distance
        # Compute squared distance from every point to every centroid, where distances[i,j] = ||X[i] - centroids[j]||²
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1) 
        # Step 2.2 : Update each centroid to the mean of its assigned points
        new_centroids = np.zeros(centroids.shape)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        # Step 2.3 : if centroids do not change, then, it means we have likely converged.
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
        
    return labels.tolist(), [[round(float(v), 4) for v in c] for c in centroids]