import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    m = np.array(m, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    param = np.array(param, dtype=np.float64)
    grad = np.array(grad, dtype=np.float64)
    # Step 1: Update first moment estimate
    m = beta1 * m + (1 - beta1) * grad
    # Step 2: Update second moment estimate
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    # Step 3: Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    # Step 4: Parameter update
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v