import torch

# Each node stores: Forward output, Backward function, References to parent nodes.
# PyTorch's default backward() computes gradients of a scalar-valued function, and .sum() converts a vector output into a scalar objective.

# forward pass builds a DAG of grad_fn nodes; backward pass traverses that DAG in reverse applying the chain rule.

# The mental model : 
# 1. Tensor = data storage + metadata(data_type, offset, shape, stride)
# 2. Autograd Node = operation + references to saved tensors

# A conceptual design would look like : 
# class Tensor:
#     def __init__(self, data):
#         self.data = data
#         self.grad = None
#         self.grad_fn = None # It means that grad_fn refers to the Node in the computation graph that produced this tensor
#         self.requires_grad = False
# And, the computation graph node will be : 
#
# class Node:
#     def __init__(self):
#         self.parents = []
#     def backward(self, grad_output):
#         raise NotImplementedError
#
# This abstract class will be implemented as operation Node for each type of operation.


# In the below example, x is a tensor, a = x ** 3, b = 2 * x, c = a + b , y = c.sum()
# x, a, b, c, y have different ids (That is, id(a) != id(b), and so on)/

# The objects in memory now look roughly like: [Tensor x, Tensor a, Tensor b, Tensor c], where each of them looks like : 
# Tensor x
#  ├── data=[1,2,3]
#  ├── grad=None
#  └── grad_fn=None
# Tensor a
#  ├── data=[1,8,27]
#  └── grad_fn=PowBackward
# Tensor b
#  ├── data=[2,4,6]
#  └── grad_fn=MulBackward
# Tensor c
#  ├── data=[3,12,33]
#  └── grad_fn=AddBackward
# Tensor y
#  ├── data=48
#  └── grad_fn=SumBackward


# The computation graph, on the other hand, looks conceptually like : 
#      SumBackward
#           |
#           c
#           |
#      AddBackward
#       /       \
#      a         b
#      |         |
# PowBackward MulBackward
#       \       /
#           x

# A more realistic graph would be like : 
# 
# Tensor y
#    |
#    v
# SumBackward
#    |
# Tensor c
#    |
#    v
# AddBackward
#   / \
#  /   \
# v     v
# Tensor a      Tensor b
#    |              |
#    v              v
# PowBackward   MulBackward
#      \          /
#       \        /
#        Tensor x
# So, in the below example, x is not literally stored "inside" the graph. 
# The graph nodes hold references to x (or to tensors derived from x) whenever those values are needed later to compute local derivatives during the backward pass.
# The graph nodes are operations, not tensors.
# Tensors carry references to graph nodes (grad_fn), while graph nodes may save references to tensors needed for backward.
# Tensors are the values flowing through the graph, and Nodes are the operations that know how to propagate gradients backward through those values.

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx

    Mathematical function:

        y = Σ (x_i³ + 2x_i)

    Therefore:

        dy/dx_i = 3x_i² + 2
    """

    # ---------------------------------------------------------
    # Step 1: Create a leaf tensor.
    #
    # x:
    #   requires_grad=True
    #   grad_fn=None (leaf node)
    #
    # Graph:
    #
    #     x
    #
    # ---------------------------------------------------------
    x = torch.tensor(
        values,
        requires_grad=True,
        dtype=torch.float32
    )

    # ---------------------------------------------------------
    # Step 2: Compute a = x³.
    #
    # Autograd creates a PowBackward node.
    #
    # Graph:
    #
    #     x
    #      \
    #       PowBackward
    #            |
    #            a
    #
    # ---------------------------------------------------------
    a = x ** 3

    # ---------------------------------------------------------
    # Step 3: Compute b = 2x.
    #
    # Autograd creates a MulBackward node.
    #
    # Graph:
    #
    #          x
    #         / \
    #        /   \
    #   Pow       Mul
    #    |         |
    #    a         b
    #
    # ---------------------------------------------------------
    b = 2 * x
    # ---------------------------------------------------------
    # Step 4: Compute c = a + b.
    #
    # Autograd creates an AddBackward node.
    #
    # Graph:
    #
    #          x
    #         / \
    #        /   \
    #      Pow    Mul
    #      |       |
    #      a       b
    #       \     /
    #       AddBackward
    #          |
    #          c
    #
    # ---------------------------------------------------------
    c = a + b
    # ---------------------------------------------------------
    # Step 5: Reduce vector output to a scalar.
    #
    # backward() expects a scalar output unless an
    # explicit upstream gradient is provided.
    #
    # y = Σ c_i
    #
    # Autograd creates a SumBackward node.
    #
    # Final graph:
    #
    #          x
    #         / \
    #        /   \
    #      Pow    Mul
    #       |      |
    #       a      b
    #        \    /
    #       AddBackward
    #            |
    #            c
    #            |
    #       SumBackward
    #            |
    #            y
    #
    # ---------------------------------------------------------
    y = c.sum()
    # ---------------------------------------------------------
    # Step 6: Reverse-mode automatic differentiation.
    #
    # Starts with:
    #
    #     dy/dy = 1
    #
    # Then traverses the graph in reverse:
    #
    #     y
    #     ↑
    #    Sum
    #     ↑
    #    Add
    #    ↑ ↑
    #  Pow Mul
    #    ↑ ↑
    #     x
    #
    # Gradients accumulated into x.grad:
    #
    #     d(x³)/dx = 3x²
    #     d(2x)/dx = 2
    #
    #     dy/dx = 3x² + 2
    #
    # ---------------------------------------------------------
    y.backward()
    # If we do y.backward() twice, then, we get error. Because, the computation graph nodes are usually no longer needed, as PyTorch releases them to save memory. The only way we can prevent this is by using y.backward(retain_graph=True)
    return x.grad.tolist()
