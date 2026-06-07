import numpy as np

class KNNClassifierV1:
    def __init__(self, k: int = 3):
        self.X_train = None 
        self.y_train = None
        self.k = k

    def fit(self, X, y):
        # In KNN, we don't have any training step.
        # All we do is just store the training set (X_train, y_train) so that they can be loaded into RAM at test-time.
        # So, training is O(1)
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.int64)

    def predict(self, X):
        # In KNN, test-time is compute-heavy.
        # Why ? Because, for each point q in X_test, we need to compute the distance of q with respect to every point in X_train.
        predictions = [ ]
        for x_test in X:
            # Step 1  : Compute distance of x_test with respect to every point in X_train
            distances = np.sum((self.X_train - x_test) ** 2, axis = 1)
            dist_indices = np.arange(distances.shape[0])
            # Step 2 : Sort by (distance, index) to enforce tie-breaking. NumPy sorts by the last key first.
            # sorted_indices = np.lexsort((dist_indices, distances))
            sorted_indices = np.argsort(distances)
            top_k_sorted_indices = sorted_indices[0: self.k]
            # Step 3 : The prediction assigned to x_test is the mean of the values attached to points in X_train whose indices are in top_k_sorted_indices 
            neighbor_labels = self.y_train[top_k_sorted_indices]
            prediction = np.bincount(neighbor_labels).argmax()
            predictions.append(prediction)
        return predictions 

class KNNClassifierV2:
    def __init__(self, k: int = 3):
        self.X_train = None 
        self.y_train = None
        self.k = k

    def fit(self, X, y):

        # Optimisation 1 : Why ascontiguousarray ? For better cache-locality, which is loved by BLAS under the hood.
        self.X_train = np.ascontiguousarray(X,dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int64)
        # Optimisation 2 : Precompute ||x||² for every training point.
        self.train_norms = np.sum(self.X_train ** 2, axis=1)

    def predict(self, X):
        predictions = []
        X = np.asarray(X, dtype=np.float32)
        for x_test in X:
            query_norm = np.sum(x_test ** 2)
            distances = (self.train_norms + query_norm - 2 * self.X_train @ x_test)
            # Optimisation 3 : O(N) partial sort instead of O(NlogN) -> O(QN) instead of O(QNlogN)
            top_k = np.argpartition(distances, self.k - 1)[:self.k]
            labels = self.y_train[top_k]

            # Implement own version of bincount , as used otherwise : prediction = np.bincount(labels).argmax()
            frequency_map = np.zeros(labels.max() + 1, dtype=np.int64)
            for value in labels:
                frequency_map[value] += 1
            predictions.append(frequency_map.argmax())

        return np.asarray(predictions)


class KNNClassifierV3:

    def __init__(self, k=3):
        # Number of nearest neighbours
        self.k = k
        # Training feature matrix
        self.X_train = None
        # Training labels
        self.y_train = None
        # Precomputed ||x||² for every training point
        self.train_norms = None

    def fit(self, X, y):
        # Store training features in contiguous float32 format.
        # Why? Better cache locality, Better SIMD utilization and Half the memory of float64
        self.X_train = np.ascontiguousarray(X, dtype=np.float32)
        # Labels must be integers for bincount()
        self.y_train = np.asarray(y, dtype=np.int64)
        # Precompute ||x||² for every training point. Shape: (N,). Why ? For Reuse across all future queries.
        self.train_norms = np.sum(self.X_train ** 2, axis=1)

    def predict(self, X):
        X_test = np.ascontiguousarray(X, dtype=np.float32)

        # Compute ||q||² for every query. Shape: (Q,)
        query_norms = np.sum(X_test ** 2, axis=1)
        # Distance formula: ||a-b||² = ||a||² + ||b||² - 2(a·b)
        # Shapes:
        #
        # query_norms[:,None]   -> (Q,1)
        # train_norms[None,:]   -> (1,N)
        # X @ X_train.T         -> (Q,N)
        #
        # Final: distances             -> (Q,N)
        distances = (query_norms[:, None] + self.train_norms[None, :] - 2 * (X @ self.X_train.T))

        predictions = []
        for row in distances:
            # Find K smallest distances. Complexity: O(N) instead ofO(N log N)
            top_k = np.argpartition(row, self.k - 1)[:self.k]
            labels = self.y_train[top_k]
            # Majority vote : Ties automatically resolve to the smallest label because argmax returns first maximum.
            prediction = np.bincount(labels).argmax()
            predictions.append(prediction)

        return np.asarray(predictions, dtype=np.int64)
        
def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    # classifier = KNNClassifierV1(k)
    # classifier = KNNClassifierV2(k)
    classifier = KNNClassifierV3(k)
    classifier.fit(X_train, y_train)
    result = classifier.predict(X_test)
    return result
