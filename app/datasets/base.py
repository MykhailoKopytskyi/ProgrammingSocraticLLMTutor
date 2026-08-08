from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..common.models import BugAnnotation


@dataclass(frozen=True)
class RawBenchmarkCase:
    """
    Dataset-normalized case before generated tests are added.
    """

    case_id: str
    problem_statement: str
    buggy_code: str
    correct_code: str
    bugs: tuple[BugAnnotation, ...]
    source: str


class DatasetLoader(ABC):
    """
    Interface implemented by every benchmark dataset loader.
    """

    @abstractmethod
    def load(
        self,
        *,
        limit: int | None = None,
    ) -> list[RawBenchmarkCase]:
        raise NotImplementedError
