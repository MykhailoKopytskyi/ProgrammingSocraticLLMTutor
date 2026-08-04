from __future__ import annotations

import re
from pathlib import Path

from ..common.models import BenchmarkCase, BugAnnotation

_TAG_PATTERN = re.compile(
    r"<(?P<tag>problem|bug_fixes|bug_desc)>\s*"
    r"(?P<body>.*?)\s*</(?P=tag)>",
    flags=re.DOTALL | re.IGNORECASE,
)

_HEADER_PATTERN = re.compile(
    r"^\s*(?:[rubfRUBF]{0,2})?"
    r"(?P<quote>\"\"\"|''')"
    r"(?P<body>.*?)"
    r"(?P=quote)",
    flags=re.DOTALL,
)


def _split_entries(
    text: str,
    expected_count: int | None = None,
) -> list[str]:
    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", text.strip())
        if part.strip()
    ]

    if expected_count is not None and len(paragraphs) == expected_count:
        return paragraphs

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if expected_count is None or len(lines) == expected_count:
        return lines

    if expected_count == 1:
        return [" ".join(text.split())]

    return lines


def parse_multi_debug_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> BenchmarkCase:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")

    header_match = _HEADER_PATTERN.match(raw)

    if header_match is None:
        raise ValueError(f"{file_path}: expected a leading annotation block")

    tags = {
        match.group("tag").lower(): match.group("body").strip()
        for match in _TAG_PATTERN.finditer(header_match.group("body"))
    }

    missing_tags = {
        "problem",
        "bug_fixes",
        "bug_desc",
    } - tags.keys()

    if missing_tags:
        raise ValueError(f"{file_path}: missing tags {sorted(missing_tags)}")

    buggy_code = raw[header_match.end() :].lstrip("\r\n")

    fixes = _split_entries(tags["bug_fixes"])

    descriptions = _split_entries(
        tags["bug_desc"],
        expected_count=len(fixes),
    )

    if len(fixes) != len(descriptions):
        raise ValueError(
            f"{file_path}: found {len(fixes)} fixes but "
            f"{len(descriptions)} descriptions"
        )

    if not 1 <= len(fixes) <= 3:
        raise ValueError(f"{file_path}: expected 1-3 bugs, found {len(fixes)}")

    bugs = [
        BugAnnotation(
            bug_id=f"bug_{index}",
            description=description,
            fix=fix,
        )
        for index, (description, fix) in enumerate(
            zip(descriptions, fixes, strict=True),
            start=1,
        )
    ]

    if root is None:
        relative_path = file_path
    else:
        relative_path = file_path.relative_to(Path(root))

    case_id = "__".join(relative_path.with_suffix("").parts)

    return BenchmarkCase(
        case_id=case_id,
        problem_statement=tags["problem"],
        buggy_code=buggy_code,
        tests="",
        student_question=(
            "My implementation does not behave as the problem "
            "describes. Can you help me identify what I have "
            "misunderstood ?"
        ),
        observed_failure="",
        bugs=bugs,
        correct_code="",
        student_misconceptions=[],
        source=f"MULTI_DEBUG:{relative_path.as_posix()}",
    )


def load_multi_debug(
    root: str | Path,
    *,
    limit: int | None = None,
) -> list[BenchmarkCase]:
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"MULTI_DEBUG directory does not exist: {root_path}")

    cases: list[BenchmarkCase] = []
    errors: list[str] = []

    for path in sorted(root_path.rglob("*.py")):
        try:
            case = parse_multi_debug_file(
                path,
                root=root_path,
            )
        except ValueError as error:
            errors.append(str(error))
            continue

        cases.append(case)

        if limit is not None and len(cases) >= limit:
            break

    if not cases:
        error_preview = "\n".join(errors[:10])

        raise ValueError(
            f"No valid MULTI_DEBUG cases found under {root_path}.\n{error_preview}"
        )

    return cases
