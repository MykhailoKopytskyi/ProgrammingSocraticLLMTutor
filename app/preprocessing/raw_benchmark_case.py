from __future__ import annotations

from dataclasses import dataclass

from ..common.models import BugAnnotation


@dataclass(frozen=True)
class RawBenchmarkCase:
    """
    Dataset-normalised case before generated tests are added.
    """

    case_id: str
    problem_statement: str
    buggy_code: str
    correct_code: str
    bugs: tuple[BugAnnotation, ...]
    source: str
