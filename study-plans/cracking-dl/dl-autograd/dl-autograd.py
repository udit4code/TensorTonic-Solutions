def autograd(operations, input_values):
    """
    Returns: Dict with "output" (float) and "gradients" (list of floats),
    rounded to 4 decimals.
    """

    class Node:
        def __init__(self, data, prev=()):
            self.data = float(data)
            self.grad = 0.0
            self.prev = prev
            self._backward = lambda: None

    nodes = [Node(v) for v in input_values]
    n_inputs = len(input_values)

    # Forward pass: build computation graph
    for op in operations:

        if op[0] == "add":
            a, b = nodes[op[1]], nodes[op[2]]

            out = Node(a.data + b.data, (a, b))

            def _backward(a=a, b=b, out=out):
                # out = a + b
                # d(out)/da = 1
                # d(out)/db = 1
                a.grad += out.grad
                b.grad += out.grad

            out._backward = _backward
            nodes.append(out)

        elif op[0] == "mul":
            a, b = nodes[op[1]], nodes[op[2]]

            out = Node(a.data * b.data, (a, b))

            def _backward(a=a, b=b, out=out):
                # out = a * b
                # d(out)/da = b
                # d(out)/db = a
                a.grad += b.data * out.grad
                b.grad += a.data * out.grad

            out._backward = _backward
            nodes.append(out)

        elif op[0] == "neg":
            a = nodes[op[1]]

            out = Node(-a.data, (a,))

            def _backward(a=a, out=out):
                # out = -a
                # d(out)/da = -1
                a.grad += -1.0 * out.grad

            out._backward = _backward
            nodes.append(out)

    output_node = nodes[-1]

    # d(output) / d(output) = 1
    output_node.grad = 1.0

    # Build topological ordering
    topo = []
    visited = set()

    def build_topo(node):
        if node not in visited:
            visited.add(node)

            for parent in node.prev:
                build_topo(parent)

            topo.append(node)

    build_topo(output_node)

    # Reverse-mode autodiff
    for node in reversed(topo):
        node._backward()

    gradients = [
        round(nodes[i].grad, 4)
        for i in range(n_inputs)
    ]

    return {
        "output": round(output_node.data, 4),
        "gradients": gradients,
    }