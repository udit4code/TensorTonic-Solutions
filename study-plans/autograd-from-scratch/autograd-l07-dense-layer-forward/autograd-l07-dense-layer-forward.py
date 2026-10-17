import torch


class DenseLayer:
    def __init__(
        self, 
        weight_matrix: torch.Tensor, 
        biases: torch.Tensor, 
        nonlinear: bool = False
    ):
        """
        Initializes the dense layer by storing weights, biases, 
        and the nonlinearity configuration as internal attributes.
        """
        self.weight = weight_matrix
        self.bias = biases
        self.nonlinear = nonlinear

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the dense layer.
        Returns an output vector in neuron order, preserving input dtype and device.
        """
        # Linear transformation: weights @ inputs + biases
        output = self.weight @ inputs + self.bias
        
        # Apply optional hyperbolic tangent nonlinearity
        if self.nonlinear:
            output = torch.tanh(output)
            
        return output


def dense_layer_forward(
    inputs: torch.Tensor, 
    weight_matrix: torch.Tensor, 
    biases: torch.Tensor, 
    nonlinear: bool
) -> torch.Tensor:
    """
    Functional wrapper that instantiates and invokes the object-oriented DenseLayer.
    """
    layer = DenseLayer(weight_matrix=weight_matrix, biases=biases, nonlinear=nonlinear)
    return layer(inputs)