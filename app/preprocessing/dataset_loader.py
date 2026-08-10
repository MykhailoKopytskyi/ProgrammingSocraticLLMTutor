from __future__ import annotations

from abc import ABC, abstractmethod

from .raw_benchmark_case import RawBenchmarkCase


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, *, limit: int | None = None) -> list[RawBenchmarkCase]: ...
