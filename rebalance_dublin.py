import json
from itertools import product
from pathlib import Path

TRAIN_PATH = Path("data/processed/intro_prog/train.jsonl")
VALIDATION_PATH = Path("data/processed/intro_prog/validation.jsonl")
TEST_PATH = Path("data/processed/intro_prog/test.jsonl")

TRAIN_TARGET = 0.789
VALIDATION_TARGET = 0.097
TEST_TARGET = 0.114


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    records = []

    for path in [TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
        for record in load_jsonl(path):
            records.append(record)

    groups = {}

    for record in records:
        source = record["source"]

        if source not in groups:
            groups[source] = []

        groups[source].append(record)

    sources = list(groups.keys())
    total = len(records)

    best_score = None
    best_assignment = None

    # 0 = train
    # 1 = validation
    # 2 = test
    for assignment in product(range(3), repeat=len(sources)):
        if 0 not in assignment:
            continue

        if 1 not in assignment:
            continue

        if 2 not in assignment:
            continue

        train_count = 0
        validation_count = 0
        test_count = 0

        for i in range(len(sources)):
            source = sources[i]
            count = len(groups[source])
            split = assignment[i]

            if split == 0:
                train_count += count
            elif split == 1:
                validation_count += count
            else:
                test_count += count

        train_ratio = train_count / total
        validation_ratio = validation_count / total
        test_ratio = test_count / total

        score = (
            abs(train_ratio - TRAIN_TARGET)
            + abs(validation_ratio - VALIDATION_TARGET)
            + abs(test_ratio - TEST_TARGET)
        )

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    train = []
    validation = []
    test = []

    print(f"\nTotal cases: {total}\n")

    for i in range(len(sources)):
        source = sources[i]
        records_for_source = groups[source]
        split = best_assignment[i]

        if split == 0:
            split_name = "TRAIN"

            for record in records_for_source:
                train.append(record)

        elif split == 1:
            split_name = "VALIDATION"

            for record in records_for_source:
                validation.append(record)

        else:
            split_name = "TEST"

            for record in records_for_source:
                test.append(record)

        print(f"{split_name:10} {source:35} {len(records_for_source)} cases")

    print("\nFinal split:")
    print(f"train:      {len(train):4} ({len(train) / total * 100:.2f}%)")
    print(f"validation: {len(validation):4} ({len(validation) / total * 100:.2f}%)")
    print(f"test:       {len(test):4} ({len(test) / total * 100:.2f}%)")

    save_jsonl(TRAIN_PATH, train)
    save_jsonl(VALIDATION_PATH, validation)
    save_jsonl(TEST_PATH, test)

    print("\nFiles rewritten.")


if __name__ == "__main__":
    main()
