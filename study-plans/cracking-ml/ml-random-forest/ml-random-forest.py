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

    def __init__(self,max_depth=5,min_samples_split=2,max_features='sqrt', rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.max_features = max_features
        self.rng = rng

    def fit(self, X, y):
        """
        Train CART tree.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.root = self.build_tree(X=X,y=y,depth=0)
        return self

    def predict(self, X):
        """
        Predict class labels for a batch.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array(
            [
                self.predict_single_sample(self.root, sample)
                for sample in X
            ]
        )

    def should_stop(self, y, depth):
        return (depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1)

    def create_leaf_node(self, y):
        return TreeNode(prediction=self.get_majority_class(y))
        
    def build_tree(self,X,y,depth):
        """
        Recursively build CART tree.
        """
        assert isinstance(X, np.ndarray), f"X : {X} must be a numpy array in build_tree"
        assert isinstance(y, np.ndarray), f"y : {y} must be a numpy array in build_tree"
        if self.should_stop(y, depth):
            # If we feel that we should stop, then, simply create a leaf node and return it.
            return self.create_leaf_node(y)
        # Otherwise, we go for the internal node.
        best_feature_idx, best_threshold, gain = self.get_best_split(X,y)
        if best_feature_idx is None or gain <= 0:
            return TreeNode(prediction=self.get_majority_class(y))
        # Select all rows in X, where the column corresponding to the feature_idx is less than the best_threshold
        left_mask = X[:, best_feature_idx] <= best_threshold
        left_child = self.build_tree(X[left_mask],y[left_mask],depth + 1)
        right_child = self.build_tree(X[~left_mask],y[~left_mask],depth + 1)
        # Create an internal node and return it
        return TreeNode(
            feature_index=best_feature_idx,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )


    def get_randomly_chosen_feature_indices(self, n_features):
        m = 1
        if isinstance(self.max_features, int) or isinstance(self.max_features, np.int64) or isinstance(self.max_features, np.int32):
            m = min(self.max_features, n_features)
        else:
            if self.max_features == 'sqrt':
                m = np.floor(np.sqrt(n_features)).astype(np.int64)
            elif self.max_features == 'log2':
                m = np.floor(np.log2(n_features)).astype(np.int64)
        
        assert self.rng != None, "rng is None at get_randomly_chosen_features"
        return self.rng.choice(n_features, size=m, replace=False)
        
    def get_best_split(self,X,y):
        """
            Find split with highest Gini gain.
        """
        n_samples, n_features = X.shape
        parent_gini = self.get_gini_impurity(y)
        best_gain = -1
        best_feature = None
        best_threshold = None

        feature_indices = self.get_randomly_chosen_feature_indices(n_features)

        for feature_idx in feature_indices:
            # By default, np.unique() returns the sorted unique elements of an array
            # So, np.unique(X[:, feature_idx]) selects the column with feature_idx across all rows, and returns a sorted list of unique values of column across the rows of X for that given feature_idx
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # Now, for each threshold, select all rows that have feature/column value for that feature_idx <= threshold.
                # Via complementary laws, select all rows that have feature values greater than threshold
                left_mask = (X[:, feature_idx] <= threshold)
                right_mask = ~left_mask

                # Count number of values in left_mask and right_mask, because we want the count of number of rows that have value less than threshold for given feature_idx and the other way round.
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left == 0 or n_right == 0:
                    # This is an edge case. So in case, all belong to same mask, then, no point splitting it.
                    continue

                left_gini = self.get_gini_impurity(y[left_mask])
                right_gini = self.get_gini_impurity(y[right_mask])
                # Computed weighted_gini across n_left and n_right
                weighted_gini = ((n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini)
                # Compute the gini gain by splitting based on threshold, w.r.t its parent gini value
                gain = (parent_gini - weighted_gini)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return (best_feature,best_threshold,best_gain)


    def predict_single_sample(self,node,sample):
        """
        Traverse tree until leaf. Similar to a tree traversal in a Binary Search Tree.
        """
        if node.is_leaf:
            return node.prediction
        if (sample[node.feature_index] <= node.threshold):
            return self.predict_single_sample(node.left,sample)
            
        return self.predict_single_sample(node.right,sample)


    def get_gini_impurity(self, y):
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

    def get_majority_class(self, y):
        """
            Return most frequent class.
        """
        classes, counts = np.unique(y,return_counts=True)
        return classes[np.argmax(counts)]


class RandomForestClassifier:
    """
        Bagging ensemble of CART classifiers.
    
        Responsibilities:
            - Generate bootstrap samples
            - Train multiple CART trees
            - Aggregate predictions via majority voting
    """
    def __init__(self,n_estimators=10,max_depth=5,min_samples_split=2,seed=42, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.trees = []
        self.max_features = max_features

    def fit(self, X, y):
        """
            Train n_estimators CART trees on bootstrap samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.trees = []
        # Observation : Each tree acts on a different bootstrapped dataset.
        # There is no communication among trees during training.
        # Isn't Training of Trees embarassingly parallel ? So, we have a scope of optimisation.
        for _ in range(self.n_estimators):
            # For each tree, get a bootstrapped training dataset and then use it to train the tree
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            tree =CARTClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features, 
                rng=self.rng
            )
            tree.fit(X_bootstrap, y_bootstrap)
            # Once the tree is trained, append it to a global list of trees
            self.trees.append(tree)
    
        return self

    def bootstrap_sample(self, X, y):
        """
            Generate one bootstrap sample.
        Returns:
            X_bootstrap, y_bootstrap
        """
        n = len(X)
        # Sample n indices with replacement
        bootstrap_indices = self.rng.randint(0, n, size=n)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        return X_bootstrap, y_bootstrap

    def predict(self, X):
        """
        Predict class labels for all samples.

        Steps:
            1. Collect prediction from every tree
            2. Perform majority vote per sample
            3. Return final predictions
        """
        X = np.asarray(X, dtype=float)
        # Shape: (n_estimators, n_test_samples)
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
    
        predictions = []
        # Iterate over test samples (columns)
        for sample_votes in all_predictions.T:
            predictions.append(self.get_majority_vote(sample_votes))
    
        return np.array(predictions)

    def get_majority_vote(self, votes):
        """
        votes:
            1D array of labels from all trees
            for a single sample

        Returns:
            winning label

        Tie-breaking:
            smallest label wins
        """
        classes, counts = np.unique(votes, return_counts=True)
        return classes[np.argmax(counts)]





def random_forest_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, max_features='sqrt', seed=42):
    """
    Returns: list of predicted class labels for each test point
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        max_features=max_features,
        seed=seed
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier.predict(X_test)
