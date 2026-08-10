import random
from collections import defaultdict


class DublinAssignmentSplitter:
    def __init__(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        seed: int = 42,
    ):
        self._train_ratio = train_ratio
        self._validation_ratio = validation_ratio
        self._seed = seed

    def split(self, records: list[dict]) -> dict[str, list[dict]]:
        by_assignment = defaultdict(list)
        for record in records:
            by_assignment[record["assignment_id"]].append(record)

        assignment_ids = sorted(by_assignment)
        rng = random.Random(self._seed)
        rng.shuffle(assignment_ids)

        train_end = int(len(assignment_ids) * self._train_ratio)
        validation_end = train_end + int(len(assignment_ids) * self._validation_ratio)
        train_ids = set(assignment_ids[:train_end])
        validation_ids = set(assignment_ids[train_end:validation_end])
        test_ids = set(assignment_ids[validation_end:])

        return {
            "train": self._collect(by_assignment, train_ids),
            "validation": self._collect(by_assignment, validation_ids),
            "test": self._collect(by_assignment, test_ids),
        }

    @staticmethod
    def _collect(by_assignment, assignment_ids):
        records = []
        for assignment_id in assignment_ids:
            for record in by_assignment[assignment_id]:
                records.append(record)
        return records
