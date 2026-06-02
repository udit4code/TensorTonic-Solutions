from abc import ABC, abstractmethod
import numpy as np


class NormalizationStrategy(ABC):

    @abstractmethod
    def compute_norm(self, matrix, axis):
        pass

class L1NormalizationStrategy(NormalizationStrategy):

    def compute_norm(self, matrix, axis):
        return np.sum(
            np.abs(matrix),
            axis=axis,
            keepdims=True
        )

class L2NormalizationStrategy(NormalizationStrategy):

    def compute_norm(self, matrix, axis):
        return np.sqrt(
            np.sum(
                matrix ** 2,
                axis=axis,
                keepdims=True
            )
        )


class MaxNormalizationStrategy(NormalizationStrategy):

    def compute_norm(self, matrix, axis):
        return np.max(
            np.abs(matrix),
            axis=axis,
            keepdims=True
        )

class NormalizationStrategyFactory:

    _strategies = {
        "l1": L1NormalizationStrategy(),
        "l2": L2NormalizationStrategy(),
        "max": MaxNormalizationStrategy()
    }

    @classmethod
    def get_strategy(cls, norm_type):
        return cls._strategies.get(norm_type)
        
def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    try:
        matrix = np.array(matrix, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2:
        return None
    if axis is not None and axis not in (0, 1):
        return None
    strategy = NormalizationStrategyFactory.get_strategy(norm_type)
    if strategy is None:
        return None
    norms = strategy.compute_norm(matrix, axis)
    return np.divide(matrix,norms, out=np.zeros_like(matrix), where=(norms != 0))