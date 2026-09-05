import math

def compute_accuracy_from_log(log: list) -> float: 
    n = len(log)
    correct_predictions = 0
    for item in log:
        if item["actual"] == item["prediction"]:
            correct_predictions += 1
    accuracy = correct_predictions / n 
    return accuracy 


def get_agreement_rate(production_log: list, shadow_log: list) -> float: 
    n = len(production_log)
    agreement_count = 0
    for prod_item, shadow_item in zip(production_log, shadow_log):
        if prod_item["prediction"] == shadow_item["prediction"]:
            agreement_count += 1
    agreement_rate = agreement_count / n 
    return agreement_rate 
    
def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    assert len(production_log) == len(shadow_log), f"production_log and shadow_log don't have same len"
    n = len(production_log)
    # Step 1 : Get Accuracy from production_log and shadow_log
    production_accuracy = compute_accuracy_from_log(production_log)
    shadow_accuracy = compute_accuracy_from_log(shadow_log)
    # Step 2 : Compute Accuracy gain
    accuracy_gain = shadow_accuracy - production_accuracy
    # Step 3 : Get agreement rate between production_log and shadow_log
    agreement_rate = get_agreement_rate(production_log, shadow_log)
    # Step 4 : Get Shadow's p95 latency
    index = math.ceil(0.95 * n) - 1 
    latencies = sorted(item["latency_ms"] for item in shadow_log)
    shadow_latency_p95 = latencies[index]

    # Step 5: Decide promotion 
    decision = accuracy_gain >= criteria["min_accuracy_gain"] and shadow_latency_p95 <= criteria["max_latency_p95"] and agreement_rate >= criteria["min_agreement_rate"]

    return {
        "promote": decision, 
        "metrics": {
            "shadow_accuracy": shadow_accuracy, 
            "production_accuracy": production_accuracy, 
            "accuracy_gain": accuracy_gain, 
            "shadow_latency_p95": shadow_latency_p95, 
            "agreement_rate": agreement_rate
        }
    }
    