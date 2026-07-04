import torch
import torch.nn as nn

class CustomSGD(torch.optim.Optimizer):
    """
    A simple implementation of SGD with optional momentum.

    step() returns:
        loss (if closure is provided) or None.
    """

    def __init__(self, params, lr=0.01, momentum=0.0):

        # Store optimizer hyperparameters.
        defaults = dict(lr=lr,momentum=momentum)
        # Initialize the base Optimizer class.
        super().__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        # Some optimizers (e.g. LBFGS) reevaluate the model by calling a closure. 
        # SGD typically ignores it, but the Optimizer API supports it.
        if closure is not None:
            loss = closure()

        # Parameter updates should not be tracked by autograd.
        with torch.no_grad():
            # Each parameter group may have different hyperparameters.
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                # Iterate over every learnable parameter in this group.
                for p in group["params"]:
                    # Skip parameters that did not receive gradients.
                    if p.grad is None:
                        continue
                    grad = p.grad
                    # Obtain the optimizer state associated with this parameter.
                    state = self.state[p]
                    # Initialize the momentum buffer on the first step.
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    # Momentum update: v = momentum * v + grad
                    buf.mul_(momentum).add_(grad)
                    # SGD update: p = p - lr * v
                    p.add_(buf, alpha=-lr)

        return loss
