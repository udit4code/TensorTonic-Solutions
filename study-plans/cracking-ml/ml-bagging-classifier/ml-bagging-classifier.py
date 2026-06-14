import numpy as np

# MOTIVATION : Why do we need a bagging classifier ? 
# In the world of only 1 decision tree, the flow is : Training Data -> Decision Tree -> Predictions.
# But, Decision trees have a major weakness: Small changes in training data can produce very different trees.
# So, they are very much sensitive to minor changes in the dataset and hence, they show high variance.
# A tiny data change produced a different model.This phenomenon is called high variance.

# So, how do we make our decision trees less sensitive to minor changes in training data ?
# We use bagging, where the central idea is based on the flow -> different training datasets -> different decision trees -> final conclusion based on a consensus (majority vote)among decisions made by all trees.
# This is based on the intuition that : One tree gives us only one opinion, but many trees can be thought of as a committee, which gives an overall decision based on opinions of individual trees. 

# But, where do we get so many training datasets for bagging ? 
# For this, we use bootstrap sampling, where given an existing training dataset, we create a new dataset by sampling WITH replacement.
# Eg : [A B C D E] -> [A B B C E] (where D is missing), [A A D D E] (where B, C are missing), [C C D E E] (where A, B are missing) 
# As we can see, some rows are repeated, while some rows are missing. The missing rows are called Out-of-bag-samples. Out-of-Bag (OOB) samples are one of the nicest side effects of bootstrap sampling.
# Since sampling is done with replacement for a given tree, some rows are repeated while some rows are missed. The unselected rows (which are missed) are known as out-of-bag samples.
# Say, there are 1 rows and there is equal probability of drawing 1 row.
# So, for a given row, the probability that it is not chosen in one draw = 1 - 1/n
# Now, for n draws, this probability multiplies -> (1 - 1/n)^n, which approaches towards 1/e = 0.368
# So, we can say that 36.8% samples are OOB and 63.2% samples appear in bootstrap set.
# Why is this useful?
# For Tree #1: Training data = Bootstrap sample
# OOB samples were never seen by Tree #1.
# Therefore: OOB samples act like test samples for that tree.
# Now, when we evaluate Tree #1 against its OOB samples, it gives an unbiased estimate of how Tree #1 performs on unseen data.
# Why is this awesome ? Normally, we split data into training_set and test_set. From training set, we carve out a validation_set.
# In case of bagging, we can use all samples for training and still estimate generalization error using OOB predictions. So, No separate validation set required.
# So, in practice, if we're already bootstrapping, we  an estimate validation accuracy without a validation set, by using Out-of-bag samples. Simply put, OOB samples are training rows that were not selected into a particular bootstrap sample, and they provide a nearly free estimate of generalization performance without needing a separate validation dataset.


# Now, what we do is we train one tree per bootstrap example. 
# So, every tree sees a slightly different view of training data.
# Tree1 ≠ Tree2 ≠ Tree3. This is what we want : trees should not be equal to each other, so that they generate variety of opinions during inference.
# Pictorially, we can depict it as : 
# Original Dataset
# │
# ├──── Bootstrap #1 → Tree1
# │
# ├──── Bootstrap #2 → Tree2
# │
# ├──── Bootstrap #3 → Tree3
# │
# └──── Bootstrap #100 → Tree100

# Why does bagging reduce variance ?
# In a simple world, where each tree is independent of each other, assume that each tree has a variance of v. Now, on average, the variance will be v/n, which is way less than v. 
# In practice, trees are correlated, so reduction isn't that dramatic, but it is still significant.
# Hence, Bagging works best when base models have High Variance (as seen in Deep Decision Trees).




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
        self.root = self.build_tree(X=X,y=y,depth=0)
        return self

    def predict(self, X):
        """
        Predict class labels for a batch.
        """
        X = np.asarray(X, dtype=float)
        return np.array(
            [
                self.predict_single_sample(self.root, sample)
                for sample in X
            ]
        )

    def build_tree(self,X,y,depth):
        """
        Recursively build CART tree.
        """
        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1):
            return TreeNode(prediction=self.get_majority_class(y))

        feature, threshold, gain = self.get_best_split(X,y)

        if feature is None or gain <= 0:
            return TreeNode(prediction=self.get_majority_class(y))

        left_mask = X[:, feature] <= threshold
        left_child = self.build_tree(X[left_mask],y[left_mask],depth + 1)
        right_child = self.build_tree(X[~left_mask],y[~left_mask],depth + 1)
        return TreeNode(feature_index=feature,threshold=threshold,left=left_child,right=right_child)


    def get_best_split(self,X,y):
        """
            Find split with highest Gini gain.
        """
        n_samples, n_features = X.shape
        parent_gini = self.get_gini_impurity(y)
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

                left_gini = self.get_gini_impurity(y[left_mask])
                right_gini = self.get_gini_impurity(y[right_mask])
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

class BaggingClassifier:
    """
        Bagging ensemble of CART classifiers.
    
        Responsibilities:
            - Generate bootstrap samples
            - Train multiple CART trees
            - Aggregate predictions via majority voting
    """
    def __init__(self,n_estimators=10,max_depth=5,min_samples_split=2,seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.trees = []

    def fit(self, X, y):
        """
            Train n_estimators CART trees on bootstrap samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.trees = []
    
        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            tree = CARTClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
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


        
def bagging_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, seed=42):
    """
    Returns: list of predicted class labels for each test point
    """
    bagging_classifier = BaggingClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=2,seed=seed)
    bagging_classifier.fit(X_train, y_train)
    return bagging_classifier.predict(X_test)
    
