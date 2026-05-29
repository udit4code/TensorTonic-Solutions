import heapq 
from datetime import datetime

def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    if not models:
        return None

    heap = []
    for model in models:
        # convert timestamp to comparable integer
        ts = int(datetime.fromisoformat(model["timestamp"]).timestamp())
        heapq.heappush(
            heap,
            (
                -model["accuracy"],   # max accuracy
                model["latency"],     # min latency
                -ts,                  # latest timestamp
                model["name"]
            )
        )

    return heapq.heappop(heap)[3]