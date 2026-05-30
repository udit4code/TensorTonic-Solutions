import numpy as np

class TreeNode:
    """
    Represents a single node in the CART tree.

    Internal Node: feature_index, threshold, left, right
    Leaf Node: prediction
    """
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,prediction=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    @property
    def is_leaf(self):
        return self.prediction is not None


class CARTClassifier:
    """
    CART Decision Tree Classifier.

    Responsibilities:
        - Train decision tree
        - Store learned tree
        - Perform predictions
    """

    def __init__(self,max_depth=5,min_samples_split=2,):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Train CART tree.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._build_tree(X=X,y=y,depth=0)

        return self

    def predict(self, X):
        """
        Predict class labels for a batch.
        """
        X = np.asarray(X, dtype=float)
        return np.array(
            [
                self._predict_one(self.root, sample)
                for sample in X
            ]
        )

    ####################################################################
    # Tree Construction
    ####################################################################
    def _build_tree(self,X,y,depth):
        """
        Recursively build CART tree.
        """
        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1):
            return TreeNode(prediction=self._majority_class(y))

        feature, threshold, gain = self._best_split(X,y)

        if feature is None or gain <= 0:
            return TreeNode(prediction=self._majority_class(y))

        left_mask = X[:, feature] <= threshold
        left_child = self._build_tree(X[left_mask],y[left_mask],depth + 1)
        right_child = self._build_tree(X[~left_mask],y[~left_mask],depth + 1)
        
        return TreeNode(feature_index=feature,threshold=threshold,left=left_child,right=right_child)

    ####################################################################
    # Split Selection
    ####################################################################

    def _best_split(self,X,y):
        """
        Find split with highest Gini gain.
        """

        n_samples, n_features = X.shape
        parent_gini = self._gini(y)
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = (X[:, feature_idx] <= threshold)
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left == 0 or n_right == 0:
                    continue

                left_gini = self._gini(y[left_mask])

                right_gini = self._gini(y[right_mask])

                weighted_gini = (
                    (n_left / n_samples) * left_gini
                    +
                    (n_right / n_samples) * right_gini
                )

                gain = (parent_gini - weighted_gini)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return (best_feature,best_threshold,best_gain)

    ####################################################################
    # Prediction
    ####################################################################

    def _predict_one(self,node,sample):
        """
        Traverse tree until leaf.
        """
        if node.is_leaf:
            return node.prediction
        if (sample[node.feature_index] <= node.threshold):
            return self._predict_one(node.left,sample)
            
        return self._predict_one(node.right,sample)

    ####################################################################
    # Utilities
    ####################################################################

    def _gini(self, y):
        """
        Gini impurity.

        Gini = 1 - Σ(p²)
        """
        n = len(y)
        if n == 0:
            return 0.0

        impurity = 1.0
        for cls in np.unique(y):
            p = np.sum(y == cls) / n
            impurity -= p * p
        return impurity

    def _majority_class(self, y):
        """
        Return most frequent class.
        """
        classes, counts = np.unique(y,return_counts=True)
        return classes[np.argmax(counts)]
        
def cart_classify(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted class labels for each test point
    """
    model = CARTClassifier(max_depth=max_depth,min_samples_split=min_samples)
    model.fit(X_train,y_train)
    return model.predict(X_test).tolist()
