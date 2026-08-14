from datasets import load_dataset

from ...common.benchmark_case_store import BenchmarkCaseStore
from .case_preprocessor import IntroProgCasePreprocessor


class IntroProgPreprocessor:
    TRAIN_ASSIGNMENTS = {
        "search_iter",
        "reverse_by_swap",
        "search_recur",
        "minimum",
    }
    VALIDATION_ASSIGNMENTS = {
        "maximum",
        "index_iter",
        "sumup",
    }
    TEST_ASSIGNMENTS = {
        "swap_keys_values",
        "reverse_iter",
        "count_letters",
    }

    def __init__(
        self,
        *,
        case_preprocessor: IntroProgCasePreprocessor,
        stores: dict[str, BenchmarkCaseStore],
        limit: int | None = None,
    ):
        self._case_preprocessor = case_preprocessor
        self._stores = stores
        self._limit = limit

    def preprocess(self) -> None:
        dataset = load_dataset("koutch/intro_prog", "dublin_repair")
        records_by_submission = {}

        for split in dataset.values():
            for record in split:
                if not record["annotation"].strip():
                    continue
                submission_id = record["submission_id"]
                if submission_id not in records_by_submission:
                    records_by_submission[submission_id] = record

        records = list(records_by_submission.values())
        splits = self._split(records)

        for split_name, split_records in splits.items():
            self._process_split(split_name, split_records)

    def _split(self, records: list[dict]) -> dict[str, list[dict]]:
        splits = {
            "train": [],
            "validation": [],
            "test": [],
        }

        for record in records:
            assignment_id = record["assignment_id"]

            if assignment_id in self.TRAIN_ASSIGNMENTS:
                splits["train"].append(record)
            elif assignment_id in self.VALIDATION_ASSIGNMENTS:
                splits["validation"].append(record)
            elif assignment_id in self.TEST_ASSIGNMENTS:
                splits["test"].append(record)
            else:
                raise ValueError(f"Unknown Dublin assignment: {assignment_id}")

        return splits

    def _process_split(self, split_name: str, records: list[dict]) -> None:
        store = self._stores[split_name]
        completed = store.completed_case_ids()
        attempted = 0

        for record in records:
            case_id = f"intro_prog__{record['submission_id']}"

            if case_id in completed:
                continue
            if self._limit is not None and attempted >= self._limit:
                break

            attempted += 1
            case = self._case_preprocessor.process(record)

            if case is None:
                print(f"FAILED: {case_id}")
                continue

            store.append(case)
            completed.add(case_id)
            print(f"accepted: {case_id}")
