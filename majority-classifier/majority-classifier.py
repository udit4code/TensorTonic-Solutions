import numpy as np

class MajorityClassifier:
    """
    A baseline classifier that breaks ties based on the first appearance in y_train.
    """
    def __init__(self, tie_strategy: str = 'first_appearance', random_state: int = None):
        self.tie_strategy = tie_strategy
        self.random_state = random_state
        self.majority_class = None

    def train(self, y_train: list):
        y_train = np.array(y_train)
        if len(y_train) == 0:
            self.majority_class = None
            return
            
        # Get unique values, their first appearance indices, and counts
        values, indices, counts = np.unique(y_train, return_index=True, return_counts=True)
        
        if self.tie_strategy == 'first_appearance':
            # Sort unique classes by their order of first appearance in y_train
            sort_idx = np.argsort(indices)
            values = values[sort_idx]
            counts = counts[sort_idx]
        elif self.tie_strategy == 'random':
            rng = np.random.default_rng(self.random_state)
            max_count = np.max(counts)
            tied_indices = np.where(counts == max_count)[0]
            self.majority_class = values[rng.choice(tied_indices)]
            return

        # Pick the class with the maximum count (first occurrence wins ties)
        max_idx = np.argmax(counts)
        self.majority_class = values[max_idx]

    def predict(self, X_test: list) -> np.ndarray:
        if self.majority_class is None or len(X_test) == 0:
            return np.array([])
        return np.full(len(X_test), self.majority_class)


def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Function wrapper that instantiates and invokes the MajorityClassifier class.
    """
    clf = MajorityClassifier(tie_strategy='first_appearance')
    clf.train(y_train)
    return clf.predict(X_test)