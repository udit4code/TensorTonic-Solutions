import numpy as np 

def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Convert inputs to NumPy arrays
    points = np.asarray(points, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=int)
    
    N, D = points.shape
    updated_centroids = []
    
    for i in range(k):
        # Step 1: Extract all data points assigned to the current cluster index 'i'
        cluster_points = points[assignments == i]
        
        # Step 2: Compute the new centroid as the mean of the cluster points
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            # Edge case: If a cluster receives no points, default to zero vector 
            # (or you could handle it by keeping the old centroid if available)
            new_centroid = np.zeros(D, dtype=np.float64)
            
        # Step 3: Append the updated centroid converted back to a list
        updated_centroids.append(new_centroid.tolist())
        
    return updated_centroids
    