def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0 
    for step_id in range(steps):
        df_by_dx = 2 * a * x + b 
        x = x - lr * df_by_dx
    return x