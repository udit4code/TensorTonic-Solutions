import numpy as np

import numpy as np
import math

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    results = {}
    for feature in sorted(train_dist.keys()):
        train = train_dist[feature]
        serve = serving_dist[feature]
        psi = 0.0
        for i in range(len(train)):
            t = train[i] + eps
            s = serve[i] + eps
            psi += (s - t) * math.log(s / t)
        psi = abs(psi)
        results[feature] = {"psi": psi, "skewed": psi >= threshold}
    return results


def detect_skew_vectorized(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    results = {}

    for feature in sorted(train_dist.keys()):
        p_train = np.asarray(train_dist[feature],dtype=np.float64)
        p_serve = np.asarray(serving_dist[feature],dtype=np.float64)

        p_train = p_train + eps
        p_serve = p_serve + eps

        psi = float(np.add.reduce((p_train - p_serve) * np.log(p_train / p_serve)))
        results[feature] = {
            "psi": psi,
            "skewed": bool(psi >= threshold)
        }

    return results