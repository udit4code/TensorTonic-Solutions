import numpy as np
from abc import ABC, abstractmethod


class ActivationStrategy(ABC):

    @abstractmethod
    def activate(self, x):
        pass

class ReLUActivation(ActivationStrategy):

    def activate(self, x):
        activation_value = np.maximum(np.zeros(x.shape), x)
        derivative_value = np.where(x > np.zeros(x.shape), 1.0,0.0)
        return activation_value, derivative_value

class SigmoidActivation(ActivationStrategy):

    def activate(self, x):
        activation_value = 1 / (1 + np.exp(-x))
        derivative_value = activation_value * (1 - activation_value)
        return activation_value, derivative_value

class SwishActivation(ActivationStrategy):

    def activate(self, x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        activation_value = x * sigmoid_x
        derivative_value = (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        return activation_value, derivative_value

class TanhActivation(ActivationStrategy):

    def activate(self, x):
        activation_value = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        derivative_value = 1 - activation_value ** 2
        return activation_value, derivative_value

class LeakyReLUActivation(ActivationStrategy):

    def activate(self, x):
        alpha = 0.01
        activation_value = x if x > 0 else alpha * x
        derivative_value = 1.0 if x > 0 else alpha
        return activation_value, derivative_value

class GeLUActivation(ActivationStrategy):

    def activate(self, x):
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = float(np.tanh(inner))
        activation_value = float(0.5 * x * (1 + t))
        sech2 = 1 - t ** 2
        inner_derivative = c * (1 + 3 * 0.044715 * x ** 2)
        derivative_value = float(0.5 * (1 + t) + 0.5 * x * sech2 * inner_derivative)
        return activation_value, derivative_value

class ActivationFactory:

    @staticmethod
    def create(activation_name):

        activation_map = {
            "relu": ReLUActivation,
            "sigmoid": SigmoidActivation,
            "tanh": TanhActivation,
            "leaky_relu": LeakyReLUActivation,
            "gelu" : GeLUActivation,
            "swish" : SwishActivation
        }

        strategy_cls = activation_map.get(
            activation_name.lower()
        )
        if strategy_cls is None:
            raise ValueError(f"Unknown activation: {activation_name}")
        return strategy_cls()

class ActivationContext:

    def __init__(self, strategy):
        self.strategy = strategy

    def execute(self, x):
        return self.strategy.activate(x)
        
def activation_functions(x, activation):
    """
    Returns: list
    """
    strategy = ActivationFactory.create(activation)
    context = ActivationContext(strategy)
    result = context.execute(np.asarray(x, dtype=np.float64))

    return result
    
