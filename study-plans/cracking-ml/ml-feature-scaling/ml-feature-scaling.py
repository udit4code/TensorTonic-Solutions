from abc import ABC, abstractmethod
import numpy as np


class ScalingStrategy(ABC):

    @abstractmethod
    def scale(self, X: np.ndarray) -> np.ndarray:
        pass


class MinMaxScalingStrategy(ScalingStrategy):

    def scale(self, X: np.ndarray) -> np.ndarray:
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)

        numerator = X - min_x
        denominator = max_x - min_x

        result = np.zeros_like(X, dtype=np.float64)

        np.divide(
            numerator,
            denominator,
            out=result,
            where=denominator != 0
        )

        return result


class StandardScalingStrategy(ScalingStrategy):

    def scale(self, X: np.ndarray) -> np.ndarray:
        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)

        numerator = X - mean_x
        denominator = std_x

        result = np.zeros_like(X, dtype=np.float64)

        np.divide(
            numerator,
            denominator,
            out=result,
            where=denominator != 0
        )

        return result


class FeatureScaler:

    def __init__(self, strategy: ScalingStrategy):
        self.strategy = strategy

    def scale(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.strategy.scale(X)

class ScalingStrategyFactory:

    @staticmethod
    def create(method: str) -> ScalingStrategy:
        strategies = {
            "minmax": MinMaxScalingStrategy,
            "standard": StandardScalingStrategy,
        }

        if method not in strategies:
            raise ValueError(f"Invalid method: {method}")

        return strategies[method]()
    
def feature_scale(X, method="minmax"):
    """
    Returns: 2D list of scaled values
    """
    strategy = ScalingStrategyFactory.create(method)
    scaler = FeatureScaler(strategy)
    return scaler.scale(X)
    