import numpy as np
from collections import deque


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications
    with Noise (DBSCAN)
    """

    NOISE = -1
    UNVISITED = -2

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.X_train = None
        self.labels = None

    def fit(self, X):
        """
        Train DBSCAN on dataset X.

        Parameters
        ----------
        X : array-like, shape (N, D)

        Returns
        -------
        labels : np.ndarray
        """

        X = np.asarray(X, dtype=np.float64)
        self.X_train = X
        n_samples = len(X)
        labels = np.full(shape=n_samples,fill_value=self.UNVISITED,dtype=np.int32)
        cluster_id = 0
        for point_idx in range(n_samples):
            # if point is already processed
            if labels[point_idx] != self.UNVISITED:
                continue
            neighbors = self.find_neighbors(point_idx)
            # if point is not a core point, then mark it as NOISE
            if not self.is_core_point(neighbors):
                labels[point_idx] = self.NOISE
                continue
            # if point is a core point, then, expand the cluster from it.
            self.expand_cluster(point_idx,neighbors,cluster_id,labels)
            cluster_id += 1

        self.labels = labels
        return labels

    def predict(self, X_new):
        """
        Assign cluster labels to unseen points.

        Strategy:
            nearest core-point within eps.

        Returns
        -------
        labels
        """

        X_new = np.asarray(X_new, dtype=np.float64)
        predictions = []
        core_points = self._get_core_points()

        for sample in X_new:
            label = self.NOISE
            best_distance = float("inf")
            for idx in core_points:
                distance = np.linalg.norm(sample - self.X_train[idx])
                if (distance <= self.eps and distance < best_distance):
                    best_distance = distance
                    label = self.labels_[idx]
            predictions.append(label)
        return np.array(predictions)


    def find_neighbors(self, point_idx):
        """
        Return indices of all points within eps distance.
        """
        point = self.X_train[point_idx]
        # distances = np.linalg.norm(self.X_train - point,axis=1)
        # We want to compute the Euclidean distance from point to every row in self.X_train.
        distances = np.sqrt(np.sum((self.X_train - point) ** 2,axis=1))
        # return np.where(distances <= self.eps)[0] : Give me the indices of all training points whose distance from the current point is at most eps
        neighbors = np.flatnonzero(distances <= self.eps)
        return neighbors

    def is_core_point(self, neighbors):
        """
        Core point: >= min_samples neighbors
        """
        return len(neighbors) >= self.min_samples

    def expand_cluster(self, point_idx, neighbors,cluster_id,labels):
        """
        BFS expansion.

        All density-reachable points
        become part of same cluster.
        """

        labels[point_idx] = cluster_id
        queue = deque(neighbors)
        while queue:
            current_idx = queue.popleft()
            # Previously marked as noise.
            # Promote to border point.
            if labels[current_idx] == self.NOISE:
                labels[current_idx] = cluster_id
            if labels[current_idx] != self.UNVISITED:
                continue
            labels[current_idx] = cluster_id
            current_neighbors = self.find_neighbors(current_idx)
            if self.is_core_point(current_neighbors):
                queue.extend(current_neighbors)

    def _get_core_points(self):
        """
        Useful for prediction.

        Returns indices of all core points.
        """
        core_points = []
        for idx in range(len(self.X_train)):
            neighbors = self._find_neighbors(idx)
            if self._is_core_point(neighbors):
                core_points.append(idx)
        return core_points
        
def dbscan(X, eps=0.5, min_samples=5):
    """
    Returns: list of integer labels (-1 for noise)
    """
    model = DBSCAN(eps=eps,min_samples=min_samples)
    return model.fit(X)
