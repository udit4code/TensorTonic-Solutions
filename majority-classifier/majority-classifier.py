import numpy as np

class MajorityClassifier:
    """
    A robust baseline classifier (ZeroR) that predicts the majority class 
    from the training data. Supports configurable tie-breaking strategies.
    """
    def __init__(self, tie_strategy: str = 'first_appearance', random_state: int = None):
        """
        Initializes the classifier.
        
        Parameters:
        - tie_strategy (str): Strategy to resolve ties when multiple classes 
          share the maximum frequency. Options:
            * 'first_appearance': Prefers the class that appeared earliest in y_train.
            * 'random': Randomly selects among tied classes.
        - random_state (int, optional): Seed for reproducibility when using 'random' tie-breaking.
        """
        self.tie_strategy = tie_strategy
        self.random_state = random_state
        self.majority_class = None

    def train(self, y_train: list):
        """
        Learns the majority class from training labels y_train.
        """
        # Convert input list to a NumPy array for efficient vectorized operations
        y_train = np.array(y_train)
        
        # Handle edge case: empty training set
        if len(y_train) == 0:
            self.majority_class = None
            return
            
        # Extract unique classes, their first appearance index in y_train, and their frequencies
        # - return_index=True tracks where each unique value first appears in the raw data
        # - return_counts=True calculates the frequency of each unique value
        values, indices, counts = np.unique(y_train, return_index=True, return_counts=True)
        
        # Apply tie-breaking strategy
        if self.tie_strategy == 'first_appearance':
            # Sort unique values chronologically based on their first appearance index in y_train
            sort_idx = np.argsort(indices)
            values = values[sort_idx]
            counts = counts[sort_idx]
            
        elif self.tie_strategy == 'random':
            # Initialize a secure NumPy random number generator
            rng = np.random.default_rng(self.random_state)
            max_count = np.max(counts)
            
            # Find all positions/indices where the class count matches the maximum frequency
            tied_indices = np.where(counts == max_count)[0]
            
            # Randomly select one of the tied classes and store it, exiting early
            self.majority_class = values[rng.choice(tied_indices)]
            return

        # For 'first_appearance' (or default), pick the first candidate index with the maximum count
        max_idx = np.argmax(counts)
        self.majority_class = values[max_idx]

    def predict(self, X_test: list) -> np.ndarray:
        """
        Predicts the learned majority class for all samples in X_test.
        Returns a 1D NumPy array.
        """
        # Return an empty array if the model hasn't been trained or the test set is empty
        if self.majority_class is None or len(X_test) == 0:
            return np.array([])
            
        # Efficiently create and return a 1D array filled entirely with the majority class
        return np.full(len(X_test), self.majority_class)


def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Functional wrapper that instantiates MajorityClassifier with 'first_appearance' 
    tie-breaking, trains on y_train, and returns predictions for X_test.
    """
    clf = MajorityClassifier(tie_strategy='first_appearance')
    clf.train(y_train)
    return clf.predict(X_test)