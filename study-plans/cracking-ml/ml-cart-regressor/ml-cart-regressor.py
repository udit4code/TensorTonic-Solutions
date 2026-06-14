import numpy as np


# Assumptions : Our CART-style regression tree evaluates splits at every observed feature value rather than at midpoints between consecutive values.
# This makes it ideal for toy-educational case of Discrete Features, while computationally intensive for production cases.
# Eg : X = [1, 3, 5, 7] And using the current splitting logic, using thresholds = np.unique(X) gives us [1, 3, 5, 7]
# In this case, the meaningful splits are : x <= 1 [1 | 3, 5, 7], 1 < x <= 3 [1, 3 | 5, 7], 3 < x <= 5 [1, 3, 5 | 7]
# and 5 < x <= 7 [1, 3, 5, 7 | ]. The last split is useless since it puts everything to the left.

# Now, Assume we have continuous values : X = [1.2341, 1.2342, 1.2343, ...] and len(X) = 10000
# So, do we make around 10000 splits based on the above approach ? No, because it is very expensive and many of the splits are redundant.
# We use the midpoint approach because a decision tree only cares about how a threshold partitions the samples into a left and right group, not about the exact threshold value itself. 
# For sorted feature values like [1, 3, 5, 7], any threshold between 1 and 3 (e.g., 1.5, 2, 2.9) produces the exact same split: {1} goes left and {3,5,7} goes right. 
# Therefore, instead of evaluating every possible threshold or every observed value, we evaluate a single representative threshold, which turns out to be the midpoint between consecutive unique values—which guarantees that each distinct partition is considered exactly once. 
# This dramatically reduces computation while producing the same set of possible splits, making it the standard approach for continuous features in production CART implementations.


# Hence, let us apply midpoint approach on the same case of X = [1, 3, 5, 7]
# For instance, we have unique_values = [1, 2, 3, 4] 
# CART wants candidate split points between observed values, not on top of observed values. 
# So, we want splits along : [1 | 2, 3, 4] -> 1.5 between 1 and 2, [1, 2 | 3, 4] -> 2.5 between 2 and 3, [1, 2, 3|4] -> 3.5 between 3 and 4. 
# Hence, we do : unique_values[:-1] = [1, 2, 3] and unique_values[1:] = [2, 3, 4] 
# Now, ([1, 2, 3] + [2, 3, 4])/2 = [3, 5, 7]/2 = [1.5, 2.5, 3.5] 
# Hence, we can use : thresholds = (unique_values[:-1] + unique_values[1:]) / 2


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


class RegressionTree:

    def __init__(self, max_depth=5, min_samples=2):
        self.root = None # Training creates the Decision Tree, which will be used for inference
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X_train, y_train): # This is used during training, which will create the tree that will be pointed at by self.root
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)
        self.root = self.build_tree(X_train, y_train, 0)
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        return np.array(
            [
                self.predict_single_test_point(sample, self.root)
                for sample in X_test
            ]
        )

    def build_tree(self, X, y, depth):
        assert isinstance(X, np.ndarray), f"X : {X} must be a numpy array in build_tree"
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in build_tree"
        if self.should_stop(y, depth):
            return self.create_leaf(y)
        else:
            best_feature_index, best_threshold, best_score = self.find_best_split(X,y)
            # Special Edge case : When due to some reason, no feature could be chosen
            if best_feature_index is None:
                return self.create_leaf(y)
            # On the basis of the best feature_idx, split X and y into (left_X, left_y) and (right_X, right_y)
            left_X,left_y,right_X,right_y = self.split_dataset(X, y, best_feature_index, best_threshold)
            # Build the children recursively 
            left_child = self.build_tree(left_X,left_y,depth + 1)
            right_child = self.build_tree(right_X,right_y,depth + 1)
            # Create an internal node and return it.
            return TreeNode(
                feature_index=best_feature_index,
                threshold=best_threshold,
                left=left_child,
                right=right_child
            )
            

    def find_best_split(self, X, y):
        '''
            best_feature: Index of the feature that produced the best split (integer).
            best_threshold: Threshold value used for the split (float).
            best_score: Impurity after the split, typically weighted variance/MSE (float).
        '''
        assert isinstance(X, np.ndarray), f"X : {X} must be a numpy array in find_best_split"
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in find_best_split"
        # For CART regression, we are minimizing the weighted variance (equivalently weighted MSE) of the two child nodes
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_reduction = 0.0
        parent_variance = self.get_variance(y)
        
        for feature_idx in range(n_features):
            # For X and a chosen feature_idx, we want all rows of X, but only the column pertaining to feature_idx
            # So, we opt for X[:, feature_idx] , where feature_idx is the feature/column selector.
            feature_values = X[:, feature_idx]
            # Get Candidate thresholds : 
            # np.unique(feature_values) removes duplicates and returns the values in sorted order.
            unique_values = np.unique(feature_values)
            if len(unique_values) <= 1:
                continue
            thresholds = unique_values

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                left_y = y[left_mask] # Get the indices of all features in y that are less than or equal to threshold
                right_y = y[right_mask] # Get the indices of all features in y that are more than threshold
                # Skip invalid splits
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                child_variance = self.get_weighted_variance(left_y,right_y)
                reduction = parent_variance - child_variance
        
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return (best_feature,best_threshold,best_reduction)

    def split_dataset(self, X, y, feature_idx, threshold):
        assert isinstance(X, np.ndarray), f"X : {X} must be a numpy array in split_dataset"
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in split_dataset"
        assert X.shape[0] == len(y), f"X and y had different count in split_dataset"
        # For given X and a given feature_idx, choose all the values of the column via feature_idx across all rows.
        # Hence, we use X[:, feature_idx]
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        left_X = X[left_mask]
        left_y = y[left_mask]
    
        right_X = X[right_mask]
        right_y = y[right_mask]

        return (left_X,left_y,right_X,right_y)

    def should_stop(self, y, depth):
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in should_stop"
        # Either, at the current node, the depth >= max_depth (Maximum tree depth reached),
        # or, the number of y samples in the current node is less than min_samples threshold mandated (Not enough samples to split further)
        # or, y has 1 unique sample while the rest of the samples are repetition of it (We have got a pure Node).
        return (depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1)

    def create_leaf(self, y):
        # The prediction stored in a leaf is the average(targets reaching leaf).
        # Why ? Because the mean minimizes squared error.
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in create_leaf"
        mean = np.mean(y)
        return TreeNode(prediction=mean)
        

    def get_variance(self, y):
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in get_variance"
        y_mean = np.mean(y)
        squared_diff = (y - y_mean) ** 2
        variance = np.sum(squared_diff) / len(y)
        return variance

    def get_weighted_variance(self,left_y,right_y):
        assert isinstance(left_y, np.ndarray), f"left_y : {left_y} must be a numpy array in get_weighted_variance"
        assert isinstance(right_y, np.ndarray), f"right_y : {right_y} must be a numpy array in get_weighted_variance"
        assert len(left_y) > 0, f"left_y : {left_y} has len 0"
        assert len(right_y) > 0, f"right_y : {right_y} has len 0"
        left_variance = self.get_variance(left_y)
        right_variance = self.get_variance(right_y)
        total_count = len(left_y) + len(right_y)
        return (len(left_y) * left_variance + len(right_y) * right_variance) / total_count

    def predict_single_test_point(self, x, node):
        # For prediction, during test-time, it is basically tree traversal similar to that of search in binary search tree.
        if node.is_leaf:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self.predict_single_test_point(x,node.left)
        else:
            return self.predict_single_test_point(x,node.right) 
        
def cart_regress(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted values rounded to 4 decimal places
    """
    decision_tree = RegressionTree(max_depth=max_depth, min_samples=min_samples)
    decision_tree.fit(X_train, y_train)
    return decision_tree.predict(X_test)
    
