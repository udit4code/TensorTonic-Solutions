from functools import cmp_to_key

def compare_models(m1, m2):
    # Higher accuracy wins
    if m1["accuracy"] != m2["accuracy"]:
        return -1 if m1["accuracy"] > m2["accuracy"] else 1
    # Lower latency wins
    if m1["latency"] != m2["latency"]:
        return -1 if m1["latency"] < m2["latency"] else 1
    # More recent timestamp wins
    if m1["timestamp"] != m2["timestamp"]:
        return -1 if m1["timestamp"] > m2["timestamp"] else 1
    return 0


def is_better(a, b):
    if a["accuracy"] != b["accuracy"]:
        return a["accuracy"] > b["accuracy"]

    if a["latency"] != b["latency"]:
        return a["latency"] < b["latency"]

    return a["timestamp"] > b["timestamp"]


def promote_model(models):
    # Time Complexity : O(n)
    best = None
    for model in models:
        if best is None or is_better(model, best):
            best = model
    return best["name"]

def promote_model_via_sorting(models):
    # Time Complexity : O(nlogn)
    """
    Decide which model version to promote to production.
    """
    best_model = sorted(
        models,
        key=cmp_to_key(compare_models)
    )[0]
    return best_model["name"]
