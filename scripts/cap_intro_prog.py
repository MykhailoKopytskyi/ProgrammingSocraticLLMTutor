import json
import re
from pathlib import Path

INPUT_DIR = Path("data/processed/intro_prog")
OUTPUT_DIR = Path("data/processed/intro_prog_capped")
MAX_PER_PROBLEM = 5


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def quality_score(case: dict) -> int:
    score = 0
    if case.get("tests"):
        score += 2
    if case.get("correct_code"):
        score += 2
    if case.get("bugs"):
        score += 2
    if case.get("observed_failure"):
        score += 2
    failure = case.get("observed_failure", "").lower()
    if "timed out" in failure:
        score -= 2
    return score


def bug_signature(case: dict) -> str:
    parts = []
    for bug in case.get("bugs", []):
        parts.append(normalize(bug.get("description", "")))
    return " | ".join(parts)


def load_cases() -> list[dict]:
    cases = []
    for split in ("train", "validation", "test"):
        path = INPUT_DIR / f"{split}.jsonl"

        with path.open(encoding="utf-8") as file:
            for line in file:
                case = json.loads(line)
                case["_original_split"] = split
                cases.append(case)
    return cases


def choose_cases(group: list[dict]) -> list[dict]:
    # Remove exact duplicate buggy submissions first.
    unique = {}
    for case in group:
        buggy_code = case["buggy_code"].strip()
        if buggy_code not in unique:
            unique[buggy_code] = case
    candidates = list(unique.values())
    candidates.sort(
        key=lambda case: (
            -quality_score(case),
            case["case_id"],
        )
    )

    selected = []
    seen_bug_signatures = set()

    # Prefer different kinds of bugs.
    for case in candidates:
        signature = bug_signature(case)
        if signature in seen_bug_signatures:
            continue
        selected.append(case)
        seen_bug_signatures.add(signature)
        if len(selected) == MAX_PER_PROBLEM:
            return selected

    # Fill remaining slots if there were not enough distinct bug types.
    for case in candidates:
        if case in selected:
            continue
        selected.append(case)
        if len(selected) == MAX_PER_PROBLEM:
            break
    return selected


def main() -> None:
    cases = load_cases()
    groups = {}
    for case in cases:
        problem = normalize(case["problem_statement"])

        if problem not in groups:
            groups[problem] = []

        groups[problem].append(case)
    selected = []

    for problem, group in groups.items():
        chosen = choose_cases(group)
        selected.extend(chosen)
        print(f"{len(group):4} -> {len(chosen)}  {chosen[0]['problem_statement'][:70]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        path = OUTPUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for case in selected:
                if case["_original_split"] != split:
                    continue
                case = dict(case)
                del case["_original_split"]
                file.write(json.dumps(case, ensure_ascii=False) + "\n")

    print()
    print(f"Original cases: {len(cases)}")
    print(f"Distinct problems: {len(groups)}")
    print(f"Selected cases: {len(selected)}")


if __name__ == "__main__":
    main()
