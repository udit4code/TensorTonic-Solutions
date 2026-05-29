from abc import ABC, abstractmethod

class MetricStrategy(ABC):
    """
    Strategy Interface
    """

    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

class ClassificationMetricStrategy(MetricStrategy):

    def compute(self, y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        accuracy = (tp + tn) / len(y_true)
        precision = (tp / (tp + fp) if tp + fp > 0 else 0.0)
        recall = (tp / (tp + fn) if tp + fn > 0 else 0.0)
        f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0else 0.0)

        return sorted([("accuracy", accuracy), ("precision", precision), ("recall", recall),("f1", f1)])

class RegressionMetricStrategy(MetricStrategy):

    def compute(self, y_true, y_pred):
        n = len(y_true)
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        rmse = (sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n) ** 0.5

        return sorted([("mae", mae),("rmse", rmse)])


class RankingMetricStrategy(MetricStrategy):

    def compute(self, y_true, y_pred):

        paired = sorted(zip(y_pred, y_true),reverse=True)
        top_3 = paired[:3]
        relevant = sum(1 for _, rel in top_3 if rel == 1)
        total_relevant = sum(1 for t in y_true if t == 1)
        
        precision_at_3 = relevant / 3
        recall_at_3 = (relevant / total_relevant if total_relevant > 0 else 0.0)

        return sorted([("precision_at_3", precision_at_3), ("recall_at_3", recall_at_3)])

class MetricCalculator:

    def __init__(self, strategy: MetricStrategy):
        self.strategy = strategy


    def calculate(self, y_true, y_pred):
        return self.strategy.compute(y_true,y_pred)
        
class MetricStrategyFactory:

    strategies = {
        "classification": ClassificationMetricStrategy(),
        "regression": RegressionMetricStrategy(),
        "ranking": RankingMetricStrategy()
    }


    @staticmethod
    def get_strategy(system_type):
        if system_type not in MetricStrategyFactory.strategies:
            raise ValueError(
                f"Unsupported system {system_type}"
            )
        return MetricStrategyFactory.strategies[system_type]
        
def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    strategy = (MetricStrategyFactory.get_strategy(system_type))
    calculator = MetricCalculator(strategy)
    return calculator.calculate(y_true, y_pred)