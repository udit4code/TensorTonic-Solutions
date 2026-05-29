import heapq 
from datetime import datetime

class ModelInfo:
    def __init__(self, name, latency, accuracy, timestamp):
        self.name = name
        self.latency = latency
        self.accuracy = accuracy
        self.timestamp = datetime.fromisoformat(timestamp)


    def __lt__(self, other):
        """
        Custom comparator.

        Return True if self should come before other
        in the heap.
        """

        # Rule 1:
        # Higher accuracy wins
        if self.accuracy != other.accuracy:
            return self.accuracy > other.accuracy

        # Rule 2:
        # Lower latency wins
        if self.latency != other.latency:
            return self.latency < other.latency

        # Rule 3:
        # Latest timestamp wins
        return self.timestamp > other.timestamp


    def __repr__(self):
        return (
            f"ModelInfo("
            f"{self.name}, "
            f"acc={self.accuracy}, "
            f"lat={self.latency}, "
            f"time={self.timestamp}"
            f")"
        )


class ModelHeap:

    def __init__(self):
        self.heap = []


    def push(self, model):
        heapq.heappush(self.heap, model)


    def pop(self):
        return heapq.heappop(self.heap)


    def peek(self):
        if not self.heap:
            return None
        return self.heap[0]


    def __len__(self):
        return len(self.heap)
        
def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    if not models:
        return None

    model_heap = ModelHeap()
    for m in models:
        model = ModelInfo(
            name=m["name"],
            latency=m["latency"],
            accuracy=m["accuracy"],
            timestamp=m["timestamp"]
        )
        model_heap.push(model)
    best_model = model_heap.peek()
    return best_model.name