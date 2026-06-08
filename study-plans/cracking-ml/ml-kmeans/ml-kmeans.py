import numpy as np

def get_nearest_centroid_labels(X, centroids, k):
    n, d = X.shape 
    # Under the hood, it does :  np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2) 
    # But, it is difficult to understand. 
    # Say, we have : X = np.array([[1, 2], [3, 4], [5, 6]])
    # and, centroids = np.array([[10, 20],[30, 40]])
    # X.shape = (3, 2) = (n, d) and centroids.shape  = (2, 2) = (k, d)
    # We want every point to be compared against every centroid. So, we have to reshape X.
    # We have to reshape X from (3, 2) to (3, 1, 2) 
    # So, we have X_reshaped = [[[1, 2]], [[3, 4]], [[5, 6]]], whose shape is (3, 1, 2)
    X_reshaped = X.reshape(n, 1, d) 
    # We want to convert centroids from (2, 2) to (1, 2, 2); so that Each centroid now has a "point slot".
    # So, now, centroids become [[[10,20], [30,40]] 
    centroids_reshaped = centroids.reshape(1, k, d)
    # By laws of broadcasting, (3, 1, 2) vs (1, 2, 2) = (3, 2, 2) with an interpretation: 3 points, 2 centroids, 2 features
    diff = X_reshaped - centroids_reshaped
    # So, now, diff[0] = For point [1, 2], its diff w.r.t [10, 20] and [30, 40] is [[-9,-18], [-29,-38]].
    # diff[1] = For point [3, 4], its diff w.r.t [10, 20] and [30, 40] is [[-7,-16], [-27,-36]].
    # diff[2] = For point [5, 6], its diff w.r.t [10, 20] and [30, 40] is [[-5, -14], [-25, -34]].
    squared = diff ** 2
    # So, squared[0] = [[81, 324] , [841, 1444]]. Similarly for others.
    # Now, squared shape = (3, 2, 2) . We want to sum up in innermost dimension, 
    # so that squared_distances[0] = [(81 + 324) (841 + 1444)] = [405,2285] . Why ? Euclidean distance between 2 Cartesian points.
    squared_distances = np.sum(squared, axis=2) # dimension : (n, k)
    # Finally, squared_distances = [[405,2285], [325,2025], [261,1781]]
    # Do we need to take sqrt ? No. Why ? sqrt(a) > sqrt(b) => a > b . 
    # Now, here, for point 0, closer is centroid 0. For point 1, closer is centroid 0. For point 2, closer is centroid 0
    # Why axis 1 ? Because for [[405,2285], [325,2025], [261,1781]], 405 vs 2285, 325 vs 2025 and 261 vs 1781, each corresponding to comparison of distance of each centroid from a given point. The 0th dimension is point index.
    # The 1st index is dimension of distance to each centroid for a chosen point
    labels = np.argmin(squared_distances, axis=1)
    return labels

    
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
        labels = get_nearest_centroid_labels(X, centroids, k) 
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