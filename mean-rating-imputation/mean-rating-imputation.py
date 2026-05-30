import numpy as np 
from abc import ABC, abstractmethod



class ImputationStrategy(ABC):

    @abstractmethod
    def impute(self, matrix: np.ndarray) -> None:
        pass

class UserMeanImputationStrategy(ImputationStrategy):
    def impute(self, matrix: np.ndarray) -> None:
        user_count = matrix.shape[0]
        for user_idx in range(user_count):
            row = matrix[user_idx]
            non_zero = row[row != 0]
            mean_rating = (
                np.mean(non_zero)
                if len(non_zero) > 0
                else 0.0
            )
            row[row == 0] = mean_rating

class ItemMeanImputationStrategy(ImputationStrategy):

    def impute(self, matrix: np.ndarray) -> None:
        feature_count = matrix.shape[1]
        for item_idx in range(feature_count):
            col = matrix[:, item_idx]
            non_zero = col[col != 0]
            mean_rating = (
                np.mean(non_zero)
                if len(non_zero) > 0
                else 0.0
            )
            col[col == 0] = mean_rating
            matrix[:, item_idx] = col

class RatingImputer:

    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy

    def run(self, ratings_matrix):
        matrix = np.array(ratings_matrix, dtype=float)
        self.strategy.impute(matrix)
        return matrix.tolist()

class ImputationStrategyFactory:

    @staticmethod
    def create(mode: str) -> ImputationStrategy:
        strategies = {
            "user": UserMeanImputationStrategy,
            "item": ItemMeanImputationStrategy
        }
        if mode not in strategies:
            raise ValueError(
                f"Unknown mode: {mode}"
            )
        return strategies[mode]()
        
def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    # Write code here
    strategy = ImputationStrategyFactory.create(mode)
    imputer = RatingImputer(strategy)
    return imputer.run(ratings_matrix)
    