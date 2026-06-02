from abc import ABC, abstractmethod


class DeduplicationStrategy(ABC):

    @abstractmethod
    def select(self, records):
        pass

class FirstStrategy(DeduplicationStrategy):
    def select(self, records):
        return records[0]

class LastStrategy(DeduplicationStrategy):
    def select(self, records):
        return records[-1]

class MostCompleteStrategy(DeduplicationStrategy):
    def select(self, records):
        best_record = None
        fewest_missing = float("inf")
        for record in records:
            missing_count = 0
            for value in record.values():
                if value is None:
                    missing_count += 1
            if missing_count < fewest_missing:
                fewest_missing = missing_count
                best_record = record
        return best_record



class StrategyFactory:

    _strategies = {
        "first": FirstStrategy(),
        "last": LastStrategy(),
        "most_complete": MostCompleteStrategy()
    }

    @classmethod
    def get_strategy(cls, strategy_name):
        return cls._strategies.get(strategy_name)


def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """

    strategy_obj = StrategyFactory.get_strategy(strategy)
    if strategy_obj is None:
        return None

    hash_map = {}
    order = []

    for record in records:
        key = tuple(record[column] for column in key_columns)
        if key not in hash_map:
            hash_map[key] = []
            order.append(key)
        hash_map[key].append(record)

    result = []
    for key in order:
        result.append(strategy_obj.select(hash_map[key]))
    return result

def deduplicate_v1(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    hash_map = {}
    order = [] # Maintains the order in which the records entered.
    for record in records:
        hash_key = tuple([record[column] for column in key_columns])
        if hash_key not in hash_map:
            hash_map[hash_key] = [ ]
            order.append(hash_key)
        hash_map[hash_key].append(record)
    result = [ ]
    for hash_key in order: 
        hash_values = hash_map[hash_key]
        if strategy == "first":
            result.append(hash_values[0])
        elif strategy == "last":
            result.append(hash_values[-1])
        elif strategy == "most_complete":
            result.append(min(hash_values, key=lambda record: sum(1 for v in record.values() if v is None)))
    return result