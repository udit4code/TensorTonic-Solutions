from abc import ABC, abstractmethod
import numpy as np


class DistanceStrategy(ABC):

    @abstractmethod
    def compute(self, x, y):
        pass

class EuclideanDistanceStrategy(DistanceStrategy):
    
    def compute(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        squared_diff = (x - y) ** 2

        return np.sqrt(np.sum(squared_diff))

class CosineDistanceStrategy(DistanceStrategy):

    @staticmethod
    def get_norm(x):
        return np.sqrt(np.sum(x ** 2))

    def compute(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        x_norm = self.get_norm(x)
        y_norm = self.get_norm(y)

        if (
            np.isclose(x_norm, 0.0, atol=1e-9)
            or np.isclose(y_norm, 0.0, atol=1e-9)
        ):
            return 0.0

        dot_product = np.sum(x * y)

        return 1 - (dot_product / (x_norm * y_norm))


class MinkowskiDistanceStrategy(DistanceStrategy):

    def __init__(self, p=2):
        self.p = p

    def compute(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        diff = np.abs(x - y) ** self.p
        total = np.sum(diff)
        return total ** (1 / self.p)

# Since Manhattan is Minkowski(p=1), reuse the implementation.
class ManhattanDistanceStrategy(MinkowskiDistanceStrategy):

    def __init__(self):
        super().__init__(p=1)

    def compute(self, x, y):
        distance = super().compute(x, y)
        return int(distance)


class ChebyshevDistanceStrategy(DistanceStrategy):

    def compute(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        diff = np.abs(x - y)

        return int(np.max(diff))


class DistanceCalculator:

    def __init__(self, strategy: DistanceStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: DistanceStrategy):
        self.strategy = strategy

    def compute(self, x, y):
        return self.strategy.compute(x, y)

class DistanceStrategyFactory:

    @staticmethod
    def create(metric, p=2):

        if metric == "euclidean":
            return EuclideanDistanceStrategy()

        if metric == "cosine":
            return CosineDistanceStrategy()

        if metric == "minkowski":
            return MinkowskiDistanceStrategy(p)

        if metric == "manhattan":
            return ManhattanDistanceStrategy()

        if metric == "chebyshev":
            return ChebyshevDistanceStrategy()

        raise ValueError(metric)

def distance_metric(x, y, metric, p=2):
    """
    Compute the distance between vectors x and y using the specified metric.
    Returns: float rounded to 4 decimal places
    """
    strategy = DistanceStrategyFactory.create(metric, p)
    calculator = DistanceCalculator(strategy)
    distance = calculator.compute(x, y)
    return distance
