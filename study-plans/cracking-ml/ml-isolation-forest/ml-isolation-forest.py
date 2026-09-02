import numpy as np
from dataclasses import dataclass 
from typing import Optional 
import math 

# In Python, a dataclass is a built-in feature introduced in Python 3.7 
# via the dataclasses module that provides a decorator to automatically generate special boilerplate methods 
# (like __init__(), __repr__(), and __eq__()) for classes that primarily exist to store data. 


@dataclass
class IsolationTreeNode:
    size: int
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["IsolationTreeNode"] = None
    right: Optional["IsolationTreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

# We can optimise it further by precomputation. 
# For example, in a global list variable, we can store c_values = [c(i) for i in range(max_samples + 1)]. 
# Then, later, for every depth + c(node.size), we can do depth + c_values[node.size] instead. 
def c(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return (2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n)



class IsolationTree:

    def __init__(self, max_depth: int, rng: np.random.RandomState):
        self.max_depth = max_depth
        self.rng = rng
        self.root = None

    def fit(self, X: np.ndarray):
        self.root = self.build(X, depth=0)
        return self

    def build(self, X: np.ndarray, depth: int) -> IsolationTreeNode:
        n_samples, n_features = X.shape

        # Base case
        if n_samples <= 1 or depth >= self.max_depth:
            return IsolationTreeNode(size=n_samples)

        # IMPORTANT:
        # Reference implementation randomly picks ANY feature.
        # It does NOT first filter out constant features.
        feature_idx = self.rng.randint(0, n_features)
        column = X[:, feature_idx]

        min_value = column.min()
        max_value = column.max()
        # If the randomly selected feature cannot split,
        # terminate this node immediately.
        if min_value == max_value:
            return IsolationTreeNode(size=n_samples)

        threshold = self.rng.uniform(min_value,max_value)
        left_mask = column < threshold
        left_X = X[left_mask]
        right_X = X[~left_mask]

        current_node = IsolationTreeNode(size=n_samples,feature_idx=feature_idx,threshold=threshold)
        # Recursively build left and right child of current_node
        current_node.left = self.build(left_X,depth + 1)
        current_node.right = self.build(right_X,depth + 1)

        return current_node

    def get_path_length(self, x: np.ndarray) -> float:
        return self.compute_path_length(x=x,node=self.root,depth=0)

    def compute_path_length(self,x: np.ndarray, node: IsolationTreeNode,depth: int) -> float:
        if node.is_leaf:
            return depth + c(node.size)

        if x[node.feature_idx] < node.threshold:
            return self.compute_path_length(x,node.left,depth + 1)
        return self.compute_path_length(x,node.right,depth + 1)


class IsolationForest:

    def __init__(self,n_estimators=100,max_samples=256,seed=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        self.trees = []
        self.sample_size = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_samples = len(X)

        self.sample_size = min(self.max_samples,n_samples)

        max_depth = int(np.ceil(np.log2(self.sample_size)))

        self.trees = []

        # n_estimators means that we build that many separate Isolation Trees. 
        # Each tree is trained on its own random subsample of up to max_samples points and makes its own random feature/threshold splits. 
        # Then, for each data point x, we compute its path length in every tree, average those path lengths, and convert that average into the final anomaly score. 
        # So if n_estimators=100, we build 100 Isolation Trees and aggregate their opinions.

        for _ in range(self.n_estimators):
            # IMPORTANT:
            # Match reference RNG behavior exactly.
            if self.sample_size < n_samples:
                indices = self.rng.choice(n_samples,size=self.sample_size,replace=False)
            else:
                indices = np.arange(n_samples)

            X_sample = X[indices]

            tree = IsolationTree(max_depth=max_depth,rng=self.rng)
            tree.fit(X_sample)
            self.trees.append(tree)

        return self

    def score_samples(self,X: np.ndarray):

        X = np.asarray(X, dtype=float)
        normalization = c(self.sample_size)

        scores = []
        for x in X:

            path_lengths = [tree.get_path_length(x) for tree in self.trees]
            mean_path_length = np.mean(path_lengths)

            # Match reference behavior for tiny datasets.
            if normalization > 0:
                score = 2.0 ** (-mean_path_length / normalization)
            else:
                score = 0.5

            scores.append(score)

        return scores


def isolation_forest(X,n_estimators=100,max_samples=256,seed=42):
    """
    Returns: list of anomaly scores rounded to 4 decimal places
    """

    X = np.asarray(X, dtype=float)
    model = IsolationForest(n_estimators=n_estimators,max_samples=max_samples,seed=seed)

    model.fit(X)
    scores = model.score_samples(X)
    return [round(float(score), 4) for score in scores]