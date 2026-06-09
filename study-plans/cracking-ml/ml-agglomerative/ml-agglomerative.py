import numpy as np
import heapq



class AgglomerativeClustering:

    def __init__(
        self,
        n_clusters=2,
        linkage="single"
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def point_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    # --------------------------
    # Linkage Functions
    # --------------------------

    def single_linkage(
        self,
        cluster1,
        cluster2
    ):
        return min(
            self.point_distance(
                self.X[i],
                self.X[j]
            )
            for i in cluster1
            for j in cluster2
        )

    def complete_linkage(
        self,
        cluster1,
        cluster2
    ):
        return max(
            self.point_distance(
                self.X[i],
                self.X[j]
            )
            for i in cluster1
            for j in cluster2
        )

    def average_linkage(
        self,
        cluster1,
        cluster2
    ):
        distances = [
            self.point_distance(
                self.X[i],
                self.X[j]
            )
            for i in cluster1
            for j in cluster2
        ]

        return np.mean(distances)

    def cluster_distance(
        self,
        cluster1,
        cluster2
    ):

        if self.linkage == "single":
            return self.single_linkage(
                cluster1,
                cluster2
            )

        if self.linkage == "complete":
            return self.complete_linkage(
                cluster1,
                cluster2
            )

        return self.average_linkage(
            cluster1,
            cluster2
        )

    # --------------------------
    # Training
    # --------------------------

    def fit(self, X):

        self.X = np.array(X, dtype=np.float64)

        n = len(X)

        clusters = {
            i: {i}
            for i in range(n)
        }

        active = set(clusters.keys())

        next_cluster_id = n

        heap = []

        # Build initial heap

        for i in range(n):
            for j in range(i + 1, n):

                dist = self.point_distance(
                    self.X[i],
                    self.X[j]
                )

                heapq.heappush(
                    heap,
                    (dist, i, j)
                )

        while len(active) > self.n_clusters:

            dist, c1, c2 = heapq.heappop(heap)

            # stale heap entry
            if (
                c1 not in active or
                c2 not in active
            ):
                continue

            merged = (
                clusters[c1]
                |
                clusters[c2]
            )

            new_cluster = next_cluster_id
            next_cluster_id += 1

            clusters[new_cluster] = merged

            active.remove(c1)
            active.remove(c2)
            active.add(new_cluster)

            # compute distance
            # from new cluster
            # to all active clusters

            for other in active:

                if other == new_cluster:
                    continue

                d = self.cluster_distance(
                    merged,
                    clusters[other]
                )

                heapq.heappush(
                    heap,
                    (
                        d,
                        new_cluster,
                        other
                    )
                )

        labels = np.zeros(n, dtype=int)

        for label, cluster_id in enumerate(active):
            for point_idx in clusters[cluster_id]:
                labels[point_idx] = label

        return labels
        
def agglomerative(X, n_clusters=2, linkage='single'):
    """
    Returns: list of integer cluster labels
    """
    model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
    return model.fit(X)
