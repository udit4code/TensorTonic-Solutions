import numpy as np


# The stump does not train itself. It knows only how to predict based on its current state.
# We use decision stumps as weak-learners.
# Each stump predicts +1 if x[j] > threshold, else -1.
# j = feature_idx of X based on which we want to make the decision on
# Polarity merely flips the sign of the prediction made by the stump.
class DecisionStump:

    def __init__(self, feature_idx, threshold, polarity):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, X):
        pred = np.ones(X.shape[0]) 
        # Find all rows in X for feature feature_idx, whose chosen feature value <= threshold
        minus_one_mask = X[:, self.feature_idx] <= self.threshold
        pred[minus_one_mask] = -1
        return self.polarity * pred
        


class AdaBoostClassifier:

    def __init__(self,n_estimators=10,seed=42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        # Here, (X, y) is the incoming training set.
        n_samples = X.shape[0]
        # Initialize uniform sample weights
        # Means, each sample has same weight in the training set X.
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Step 1: Find best stump under current weights
            stump, error = self.find_best_stump(X, y, sample_weights)
            # Step 2: Compute stump weight
            alpha = self.compute_alpha(error)
            # Step 3: Predict training data
            pred = stump.predict(X)
            # Step 4: Update sample weights
            sample_weights = self.update_weights(sample_weights, y, pred,alpha)
            # Step 5: Save weak learner
            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
    
        for alpha, stump in zip(self.alphas,self.stumps):
            scores += alpha * stump.predict(X)

        return np.where(scores >= 0, 1, -1)

    # Given (X, y, sample_weights), we need to find the feature, threshold and polarity
    # such that weighted_error is minimized.
    def find_best_stump(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        best_error = float("inf")
        best_stump = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
    
            for threshold in thresholds:
                for polarity in [+1, -1]:
                    stump = DecisionStump(feature_idx=feature_idx, threshold=threshold,polarity=polarity)
    
                    pred = stump.predict(X)
                    error = np.sum(sample_weights[pred != y])
    
                    if error < best_error:
                        best_error = error
                        best_stump = stump
    
        return best_stump, best_error

    def compute_alpha(self,error):
        error = np.clip(error,1e-10,1 - 1e-10)
        return 0.5 * np.log((1 - error) / error)

    def update_weights(self,weights,y,pred,alpha):
        weights = weights * np.exp(-alpha * y * pred)
        weights /= np.sum(weights)
        return weights
        
def adaboost_classify(X_train, y_train, X_test, n_estimators=10, seed=42):
    """
    Returns: list of predicted labels in {-1, +1} for each test point
    """
    classifier = AdaBoostClassifier(n_estimators=n_estimators,seed=seed)
    # Get training set (X_train, y_train)
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    classifier.fit(X_train, y_train)
    # Get Test set (X_test)
    X_test = np.array(X_test, dtype=np.float64)
    return classifier.predict(X_test)

    
